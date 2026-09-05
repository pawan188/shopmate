"""
Razorpay TEST-mode payment client. Checkout is validator-gated and the client refuses to run with non-test keys.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_BACKEND_DIR / ".env")

from catalog.catalog import get_product_by_id  # noqa: E402
from validator.validator import validate_purchase  # noqa: E402

RAZORPAY_API = "https://api.razorpay.com/v1"


def _credentials() -> tuple[str, str]:
    key_id = os.environ.get("RAZORPAY_KEY_ID", "")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise RuntimeError(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET missing. Add TEST-mode keys "
            "to backend/.env (rzp_test_...)."
        )
    if not key_id.startswith("rzp_test_"):
        raise RuntimeError(
            f"RAZORPAY_KEY_ID does not start with rzp_test_ "
            f"(got prefix '{key_id[:9]}') - refusing to use non-test keys."
        )
    return key_id, key_secret


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    key_id, key_secret = _credentials()
    response = httpx.post(
        RAZORPAY_API + path, json=payload, auth=(key_id, key_secret), timeout=20
    )
    response.raise_for_status()
    return response.json()


def _get(path: str) -> dict[str, Any]:
    key_id, key_secret = _credentials()
    response = httpx.get(RAZORPAY_API + path, auth=(key_id, key_secret), timeout=20)
    response.raise_for_status()
    return response.json()


def create_payment_link(
    *,
    amount_inr: int,
    description: str,
    notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a hosted Razorpay TEST payment link for amount_inr (INR int)."""
    payload = {
        "amount": int(amount_inr) * 100,  # INR -> paise
        "currency": "INR",
        "accept_partial": False,
        "description": description[:255],
        "customer": {
            "name": "Hackathon Test Customer",
            "email": "hackathon.test@example.com",
            "contact": "+919000000000",
        },
        "notify": {"email": False, "sms": False},
        "notes": notes or {},
    }
    return _post("/payment_links", payload)


def fetch_payment_link(link_id: str) -> dict[str, Any]:
    """Poll the status of a payment link: created / paid / cancelled / expired."""
    return _get(f"/payment_links/{link_id}")


def checkout(
    product_id: str,
    price: Any,
    last_search_results: dict[str, Any] | None,
    audit: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """The ONLY entry point that creates a payment. Validator gate first.

    Returns either:
      {"stage": "blocked_by_validator", "validator": {...}, "payment_link": None}
    or
      {"stage": "payment_link_created", "validator": {...},
       "payment_link": {"id", "short_url", "status", "amount_paise", ...}}
    """
    decision = validate_purchase(product_id, price, last_search_results)
    if audit is not None:
        audit.log(
            "validator_decision",
            session_id=session_id or "unspecified",
            data={
                "product_id": product_id,
                "claimed_price": decision["claimed_price"],
                "retrieved_price": decision["retrieved_price"],
                "catalog_price": decision["catalog_price"],
                "decision": decision["decision"],
                "failures": decision["failures"],
            },
        )
    if decision["decision"] != "ALLOW":
        return {
            "stage": "blocked_by_validator",
            "validator": decision,
            "payment_link": None,
        }

    product = get_product_by_id(product_id)
    amount_inr = decision["catalog_price"]
    link = create_payment_link(
        amount_inr=amount_inr,
        description=f"{product['name']} - {product['category']}",
        notes={
            "product_id": product_id,
            "catalog_price_inr": amount_inr,
            "source": "grounded_checkout",
        },
    )
    payment_link_summary = {
        "id": link.get("id"),
        "short_url": link.get("short_url"),
        "status": link.get("status"),
        "amount_paise": link.get("amount"),
        "currency": link.get("currency"),
    }
    if audit is not None:
        audit.log(
            "razorpay_response",
            session_id=session_id or "unspecified",
            data={"payment_link": payment_link_summary},
        )
    return {
        "stage": "payment_link_created",
        "validator": decision,
        "payment_link": payment_link_summary,
    }


def checkout_cart(
    cart_items: list[dict[str, Any]],
    audit: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create ONE Razorpay TEST payment link for a multi-item cart.

    Every line's price is re-verified against the live catalog at payment time
    - never trusted from when it was added. Any mismatch blocks the whole cart
    before Razorpay is contacted.
    """
    failures: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for item in cart_items:
        product = get_product_by_id(item.get("product_id"))
        stored = int(item.get("price") or 0)
        if product is None:
            failures.append(
                {"product_id": item.get("product_id"), "reason": "unknown_product"}
            )
        elif int(product["price"]) != stored:
            failures.append(
                {
                    "product_id": item.get("product_id"),
                    "claimed_price": stored,
                    "catalog_price": product["price"],
                    "reason": "price_mismatch",
                }
            )
        else:
            verified.append(
                {
                    "product_id": item["product_id"],
                    "name": product["name"],
                    "price": product["price"],
                    "qty": int(item.get("qty") or 1),
                }
            )

    total_inr = sum(v["price"] * v["qty"] for v in verified)
    decision = "ALLOW" if verified and not failures else "BLOCK"
    if audit is not None:
        audit.log(
            "validator_decision",
            session_id=session_id or "unspecified",
            data={
                "action": "cart_checkout",
                "line_items": verified,
                "failures": failures,
                "decision": decision,
                "total_inr": total_inr,
            },
        )
    if decision != "ALLOW":
        return {
            "stage": "blocked_by_validator",
            "validator": {"decision": decision, "failures": failures},
            "payment_link": None,
        }

    names = ", ".join(f"{v['name']} x{v['qty']}" for v in verified)[:240]
    link = create_payment_link(
        amount_inr=total_inr,
        description=f"Cart ({len(verified)} line items) - {names}",
        notes={
            "items": verified,
            "total_inr": total_inr,
            "source": "cart_checkout",
        },
    )
    payment_link_summary = {
        "id": link.get("id"),
        "short_url": link.get("short_url"),
        "status": link.get("status"),
        "amount_paise": link.get("amount"),
        "currency": link.get("currency"),
    }
    if audit is not None:
        audit.log(
            "razorpay_response",
            session_id=session_id or "unspecified",
            data={"action": "cart_checkout", "payment_link": payment_link_summary},
        )
    return {
        "stage": "payment_link_created",
        "validator": {"decision": decision, "failures": failures},
        "payment_link": payment_link_summary,
        "total_inr": total_inr,
        "items": verified,
    }
