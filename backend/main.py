"""
FastAPI application: session-aware chat API with a grounded agent, deterministic purchase/cart intents, payment-status checks, and an audit-log view. Serves the frontend statically.
"""

from __future__ import annotations

import sys
import re
import uuid
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent.agent import ShoppingAgent  # noqa: E402
from logging_utils.audit import AuditLogger  # noqa: E402
from razorpay.client import checkout, checkout_cart, fetch_payment_link  # noqa: E402
from retrieval.search import search_catalog  # noqa: E402
from validator.validator import validate_purchase  # noqa: E402

FRONTEND_DIR = _BACKEND_DIR.parent / "frontend"

app = FastAPI(title="ShopMate - Grounded Agentic Commerce")
sessions: dict[str, ShoppingAgent] = {}
audit = AuditLogger()

BUY_KEYWORDS = (
    "buy", "checkout", "purchase", "pay for", "payment link",
    "order", "get it", "take it", "proceed",
)

# Phrases that mean the user is telling us they paid / asking us to confirm.
PAY_STATUS_PHRASES = (
    "paid", "payment done", "payment status", "confirm payment",
    "confirm the payment", "completed the payment", "done the payment",
    "payment successful", "receive my payment", "received my payment",
    "payment went through", "payment completed", "been made", "went through",
    "did you receive", "did the payment",
)

# Words that, combined with the word "payment", signal the user is asking
# whether their payment went through (not a general question about paying).
PAY_STATUS_HINT_WORDS = (
    "check", "made", "status", "confirm", "verify", "received", "receive",
    "went", "complete", "successful", "done", "track",
)

# Phrases that trigger a clearly-marked SIMULATED payment decline (demo hook).
PAY_DECLINE_TRIGGERS = (
    "declin", "payment fail", "payment unsuccessful",
    "could not complete the payment", "payment was not successful",
)

# Words that do not help identify WHICH product the user means.
_FILLER = {
    "this", "that", "please", "would", "like", "want", "need",
    "show", "tell", "give", "some", "with", "and", "for",
}


def _significant_words(message: str) -> list[str]:
    """Words in the message that could name/describe a product."""
    words = re.findall(r"[a-zA-Z0-9]{4,}", message.lower())
    return [w for w in words if w not in BUY_KEYWORDS and w not in _FILLER]


def _row_matches(row: dict, message: str) -> bool:
    """Does the message name this product? (word-overlap against its name)"""
    msg_words = set(_significant_words(message))
    name_words = set(re.findall(r"[a-z0-9]{4,}", (row.get("name") or "").lower()))
    return bool(msg_words & name_words)


def _names_any_product(message: str, agent: ShoppingAgent) -> bool:
    """Does the message name a product we know (last search or cart)?"""
    results = ((agent.last_search_results or {}).get("results")) or []
    for row in results:
        if _row_matches(row, message):
            return True
    for item in agent.cart:
        if _row_matches(item, message):
            return True
    return False


def _wants_payment_status(message: str) -> bool:
    """Is the user telling us they paid, or asking whether payment went through?"""
    lowered = message.lower()
    if any(phrase in lowered for phrase in PAY_STATUS_PHRASES):
        return True
    return "payment" in lowered and any(
        word in lowered for word in PAY_STATUS_HINT_WORDS
    )


def _wants_decline_simulation(message: str) -> bool:
    """Did the user ask to simulate a payment decline (for the demo)?"""
    lowered = message.lower()
    return any(trigger in lowered for trigger in PAY_DECLINE_TRIGGERS)


def _handle_decline_simulation(
    agent: ShoppingAgent, session_id: str
) -> dict[str, Any]:
    last_checkout = agent.last_checkout
    if last_checkout is None:
        return {
            "session_id": session_id,
            "reply": (
                "There's no payment link in this session to decline. Create one "
                "first (checkout a product or cart), then ask me to simulate a "
                "decline."
            ),
            "guard": None,
            "checkout": None,
            "cart": _cart_snapshot(agent),
        }
    audit.log(
        "payment_decline",
        session_id=session_id,
        data={
            "link_id": last_checkout["link_id"],
            "simulated": True,
            "reason": "demo_decline",
            "razorpay_called": False,
            "cart_count": sum(int(i.get("qty") or 1) for i in agent.cart),
            "cart_total_inr": sum(
                int(i.get("price") or 0) * int(i.get("qty") or 1)
                for i in agent.cart
            ),
        },
    )
    reply = (
        f"Payment DECLINED (simulated, for the demo). Link "
        f"{last_checkout['link_id']} was NOT charged and Razorpay was not "
        "called again - I never auto-retry a declined payment, to avoid "
        "double charges.\n\n"
        "Your cart is kept exactly as it was. You can retry with the same "
        "open link, modify the cart, or clear it."
    )
    return {
        "session_id": session_id,
        "reply": reply,
        "guard": None,
        "checkout": {
            "action": "decline_simulated",
            "link_id": last_checkout["link_id"],
            "simulated": True,
        },
        "cart": _cart_snapshot(agent),
    }


def _find_named_in_results(
    message: str, results: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """Return the product row the message names, if it is in these results."""
    for row in results or []:
        if _row_matches(row, message):
            return row
    return None


def _resolve_checkout_target(
    message: str, agent: ShoppingAgent
) -> tuple[dict | None, dict | None]:
    """Figure out WHICH product the user wants to buy.

    Preference order:
      1. a product the message names, if present in the session's last search;
      2. a product the message names, via a fresh catalog search;
      3. for bare confirmations like 'buy it' (no product words), the product
         the user most recently selected / the assistant just talked about
         (agent.selected_product) - grounded afresh before checkout.
    Returns (product_row, search_result_containing_it) or (None, None) when the
    user named something we cannot ground, or a bare 'buy it' with no current
    selection - we never guess.
    """
    last = agent.last_search_results
    named = bool(_significant_words(message))

    if named:
        row = _find_named_in_results(message, (last or {}).get("results") or [])
        if row is not None:
            return row, last
        # Named something not in the last results -> search the catalog for it.
        fresh = search_catalog(message, n_results=5)
        row = _find_named_in_results(message, fresh.get("results") or [])
        if row is not None:
            return row, fresh
        return None, None

    # Bare confirmation ('buy it'): the product we were just talking about.
    selected = agent.selected_product
    if selected is not None:
        if last and last.get("results"):
            for row in last["results"]:
                if row.get("product_id") == selected["product_id"]:
                    return row, last
        # Selected earlier but not in the current results -> ground it afresh.
        fresh = search_catalog(selected.get("name") or "", n_results=5)
        for row in fresh.get("results", []):
            if row.get("product_id") == selected["product_id"]:
                return row, fresh

    return None, None


def _update_selection_from_reply(
    agent: ShoppingAgent, final_text: str | None
) -> None:
    """Remember which product the assistant just talked about.

    If the grounded reply names exactly one catalog product, treat that as the
    user's current selection so a later bare 'buy it' is deterministic. If it
    names several, the reference is ambiguous - clear the selection rather
    than ever guessing.
    """
    last = agent.last_search_results
    if not last or not last.get("results") or not final_text:
        return
    text_lower = final_text.lower()
    matched = [
        row for row in last["results"]
        if (row.get("name") or "").lower() in text_lower
    ]
    if len(matched) == 1:
        agent.selected_product = matched[0]
    elif len(matched) > 1:
        agent.selected_product = None


def _cart_snapshot(agent: ShoppingAgent) -> dict[str, Any]:
    """Compact, serialisable view of the session cart."""
    items = agent.cart
    count = sum(int(i.get("qty") or 1) for i in items)
    total = sum(int(i.get("price") or 0) * int(i.get("qty") or 1) for i in items)
    return {
        "count": count,
        "total_inr": total,
        "items": [
            {
                "product_id": i["product_id"],
                "name": i["name"],
                "price": i["price"],
                "qty": i["qty"],
            }
            for i in items
        ],
    }


def _cart_response(
    session_id: str,
    agent: ShoppingAgent,
    reply: str,
    checkout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "reply": reply,
        "guard": None,
        "checkout": checkout,
        "cart": _cart_snapshot(agent),
    }


def _cart_lines(agent: ShoppingAgent) -> str:
    lines = []
    for idx, item in enumerate(agent.cart, start=1):
        unit = int(item["price"]) * int(item["qty"])
        lines.append(
            f"{idx}. {item['name']} - Rs.{item['price']} x {item['qty']} = Rs.{unit}"
        )
    return "\n".join(lines)


def _handle_cart_add(
    message: str, agent: ShoppingAgent, session_id: str
) -> dict[str, Any]:
    target, source = _resolve_checkout_target(message, agent)
    if target is None:
        return _cart_response(
            session_id,
            agent,
            "Which product should I add? Name it (e.g. 'the AirPods Pro') or ask "
            "me to show options first - I won't guess.",
        )
    decision = validate_purchase(target["product_id"], target["price"], source)
    audit.log(
        "validator_decision",
        session_id=session_id,
        data={
            "action": "add_to_cart",
            "product_id": target["product_id"],
            "claimed_price": decision["claimed_price"],
            "retrieved_price": decision["retrieved_price"],
            "catalog_price": decision["catalog_price"],
            "decision": decision["decision"],
            "failures": decision["failures"],
        },
    )
    if decision["decision"] != "ALLOW":
        return _cart_response(
            session_id,
            agent,
            "I couldn't add that to your cart - the validator blocked it because "
            "the product/price didn't match what I retrieved from the catalog. "
            "Nothing was added.",
        )
    existing = next(
        (i for i in agent.cart if i["product_id"] == target["product_id"]), None
    )
    if existing is not None:
        existing["qty"] += 1
    else:
        agent.cart.append(
            {
                "product_id": target["product_id"],
                "name": target["name"],
                "price": target["price"],
                "qty": 1,
            }
        )
    agent.selected_product = target
    snap = _cart_snapshot(agent)
    audit.log(
        "cart_add",
        session_id=session_id,
        data={
            "product_id": target["product_id"],
            "name": target["name"],
            "price": target["price"],
            "cart_count": snap["count"],
            "cart_total_inr": snap["total_inr"],
        },
    )
    reply = (
        f"Added {target['name']} (Rs.{target['price']}, exact catalog price) to "
        f"your cart. Cart now has {snap['count']} item(s), total Rs."
        f"{snap['total_inr']}.\n\n"
        "Say 'what's in my cart' to review, or 'checkout my cart' when ready."
    )
    return _cart_response(session_id, agent, reply)


def _handle_cart_remove(
    message: str, agent: ShoppingAgent, session_id: str
) -> dict[str, Any]:
    if not agent.cart:
        return _cart_response(session_id, agent, "Your cart is already empty.")
    idx = None
    for i, item in enumerate(agent.cart):
        if _row_matches(item, message):
            idx = i
            break
    if idx is None and agent.selected_product is not None:
        for i, item in enumerate(agent.cart):
            if item["product_id"] == agent.selected_product["product_id"]:
                idx = i
                break
    if idx is None and len(agent.cart) == 1:
        idx = 0
    if idx is None:
        return _cart_response(
            session_id,
            agent,
            "Which item should I remove? Your cart has:\n" + _cart_lines(agent),
        )
    removed = agent.cart.pop(idx)
    snap = _cart_snapshot(agent)
    audit.log(
        "cart_remove",
        session_id=session_id,
        data={
            "product_id": removed["product_id"],
            "name": removed["name"],
            "cart_count": snap["count"],
            "cart_total_inr": snap["total_inr"],
        },
    )
    return _cart_response(
        session_id,
        agent,
        f"Removed {removed['name']} from your cart. Cart now has {snap['count']} "
        f"item(s), total Rs.{snap['total_inr']}.",
    )


def _handle_cart_show(agent: ShoppingAgent, session_id: str) -> dict[str, Any]:
    if not agent.cart:
        return _cart_response(
            session_id,
            agent,
            "Your cart is empty. Ask me to show you products, then say "
            "'add the <product> to my cart'.",
        )
    snap = _cart_snapshot(agent)
    reply = (
        f"Your cart ({snap['count']} item(s), total Rs.{snap['total_inr']}):\n"
        + _cart_lines(agent)
        + "\n\nEvery line price was validated against the catalog. Say "
        "'checkout my cart' when you're ready."
    )
    return _cart_response(session_id, agent, reply)


def _cart_signature(agent: ShoppingAgent) -> list[tuple[str, int]]:
    """Stable fingerprint of the cart, to detect changes across checkouts."""
    return [
        (item["product_id"], int(item.get("qty") or 1)) for item in agent.cart
    ]


def _handle_cart_checkout(
    agent: ShoppingAgent, session_id: str
) -> dict[str, Any]:
    if not agent.cart:
        return _cart_response(
            session_id,
            agent,
            "Your cart is empty - nothing to check out. Add a product first "
            "(e.g. 'add the AirPods Pro to my cart').",
        )
    signature = _cart_signature(agent)
    pending = agent.pending_checkout
    if (
        pending is not None
        and pending.get("signature") == signature
        and pending.get("link_id")
    ):
        # Same, unchanged cart already has an open TEST link -> reuse it.
        # Never create a second link for the same cart (no double charge).
        agent.last_checkout = {
            "link_id": pending["link_id"],
            "product_id": None,
            "name": pending["name"],
            "price": pending["total_inr"],
            "short_url": pending["short_url"],
        }
        return _cart_response(
            session_id,
            agent,
            "Your cart hasn't changed and you already have an open Razorpay "
            "TEST payment link for it:\n"
            f"{pending['short_url']}\n\n"
            "Pay it (Netbanking, any bank) or ask to modify your cart. Your "
            "cart is kept until the payment succeeds - tell me 'I've paid' "
            "once done and I'll confirm and clear it.",
            checkout={
                "link_id": pending["link_id"],
                "short_url": pending["short_url"],
                "total_inr": pending["total_inr"],
                "count": pending["count"],
            },
        )
    unit_count = sum(int(i.get("qty") or 1) for i in agent.cart)
    result = checkout_cart(list(agent.cart), audit=audit, session_id=session_id)
    if result["stage"] == "blocked_by_validator":
        return _cart_response(
            session_id,
            agent,
            "Cart checkout BLOCKED by the validator - a stored price no longer "
            "matches the live catalog, so no payment link was created and "
            "nothing was charged. Your cart is untouched.",
        )
    link = result["payment_link"]
    total = result["total_inr"]
    agent.last_checkout = {
        "link_id": link["id"],
        "product_id": None,
        "name": f"Cart ({unit_count} items)",
        "price": total,
        "short_url": link["short_url"],
    }
    agent.pending_checkout = {
        "signature": signature,
        "link_id": link["id"],
        "short_url": link["short_url"],
        "total_inr": total,
        "name": f"Cart ({unit_count} items)",
        "count": unit_count,
    }
    reply = (
        f"Payment link created in Razorpay TEST mode for your cart "
        f"({unit_count} item(s), Rs.{total} - sum of validated catalog "
        f"prices).\n\nPay here: {link['short_url']}\n\n"
        "Your cart is kept until the payment succeeds - if you don't finish "
        "paying, the items stay here. Tell me 'I've paid' once done and I'll "
        "confirm it and clear your cart."
    )
    return _cart_response(
        session_id,
        agent,
        reply,
        checkout={
            "link_id": link["id"],
            "short_url": link["short_url"],
            "total_inr": total,
            "count": unit_count,
        },
    )


def _handle_cart(
    message: str, agent: ShoppingAgent, session_id: str
) -> dict[str, Any]:
    lowered = message.lower()
    audit.log(
        "query_received",
        session_id=session_id,
        data={"query": message, "intent": "cart"},
    )
    checkout_words = ("checkout", "pay for", "pay my cart", "order", "purchase")
    if any(w in lowered for w in checkout_words) or (
        "buy" in lowered and "cart" in lowered
    ):
        return _handle_cart_checkout(agent, session_id)
    if any(w in lowered for w in ("add", "put", "include")):
        return _handle_cart_add(message, agent, session_id)
    if any(w in lowered for w in ("remove", "delete", "drop")):
        return _handle_cart_remove(message, agent, session_id)
    if any(w in lowered for w in ("clear", "empty")):
        agent.cart = []
        agent.pending_checkout = None
        agent.last_checkout = None
        return _cart_response(
            session_id, agent, "Cleared your cart - it's empty now."
        )
    return _handle_cart_show(agent, session_id)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _get_agent(session_id: str) -> ShoppingAgent:
    agent = sessions.get(session_id)
    if agent is None:
        agent = ShoppingAgent(audit=audit, session_id=session_id)
        sessions[session_id] = agent
    return agent


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "active_sessions": len(sessions)}


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    message = (body.message or "").strip()
    session_id = body.session_id or uuid.uuid4().hex

    if not message:
        return {
            "session_id": session_id,
            "reply": "Please type a message.",
            "guard": None,
            "checkout": None,
        }

    agent = _get_agent(session_id)
    last = agent.last_search_results

    # Cart messages are handled deterministically, before single-product buy.
    if "cart" in message.lower() or "basket" in message.lower():
        return _handle_cart(message, agent, session_id)

    wants_checkout = any(k in message.lower() for k in BUY_KEYWORDS)

    # Bare checkout words ("checkout", "pay now", "proceed") with a non-empty
    # cart and no product named -> the user means the whole cart, not the last
    # item they discussed.
    if wants_checkout and agent.cart and not _names_any_product(message, agent):
        bare_checkout_words = (
            "checkout", "pay for", "pay now", "proceed", "place order",
            "place my order", "order now", "buy everything",
        )
        if any(w in message.lower() for w in bare_checkout_words):
            audit.log(
                "query_received",
                session_id=session_id,
                data={"query": message, "intent": "cart_bare_checkout"},
            )
            return _handle_cart_checkout(agent, session_id)

    # Deterministic purchase path: resolve WHICH product the user means.
    if wants_checkout:
        target, source = _resolve_checkout_target(message, agent)
        if target is None:
            audit.log("query_received", session_id=session_id, data={"query": message})
            return {
                "session_id": session_id,
                "reply": "Which product would you like to buy? Name it (e.g. 'the "
                         "Sony WH-1000XM5' or 'AirPods Pro') or ask me to show you "
                         "options first - I won't guess.",
                "guard": None,
                "checkout": None,
            }
        # Bare confirmations ('buy it') still need a confident, grounded result.
        if not _significant_words(message) and source.get("low_confidence"):
            audit.log("query_received", session_id=session_id, data={"query": message})
            return {
                "session_id": session_id,
                "reply": "I'm not confident which product you mean yet - could you "
                         "name it? I don't want to check out the wrong thing.",
                "guard": {"overridden": True, "reason": "low_confidence_buy"},
                "checkout": None,
            }
        audit.log("query_received", session_id=session_id, data={"query": message})
        try:
            result = checkout(
                target["product_id"], target["price"], source,
                audit=audit, session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001 - Razorpay/network hiccup
            audit.log(
                "razorpay_error",
                session_id=session_id,
                data={"action": "create_payment_link", "error": str(exc)[:300]},
            )
            return {
                "session_id": session_id,
                "reply": (
                    "I couldn't reach Razorpay to create the payment link right "
                    "now - nothing was charged. Please try again in a moment."
                ),
                "guard": None,
                "checkout": None,
            }
        if result["stage"] == "payment_link_created":
            link = result["payment_link"]
            agent.last_checkout = {
                "link_id": link["id"],
                "product_id": target["product_id"],
                "name": target["name"],
                "price": target["price"],
                "short_url": link["short_url"],
            }
            reply = (
                f"Payment link created in Razorpay TEST mode for {target['name']} "
                f"(Rs.{target['price']}, exact catalog price).\n\n"
                f"Pay here: {link['short_url']}\n\n"
                "Test flow: choose Netbanking and pick any bank, or use card "
                "4111 1111 1111 1111 (may need International Cards enabled). "
                "Tell me once you've paid and I'll confirm it."
            )
            return {
                "session_id": session_id,
                "reply": reply,
                "guard": None,
                "checkout": {
                    "product_id": target["product_id"],
                    "name": target["name"],
                    "price": target["price"],
                    "link_id": link["id"],
                    "short_url": link["short_url"],
                },
            }
        return {
            "session_id": session_id,
            "reply": "I couldn't create the payment link - the checkout was blocked "
                     "by the validator, so no payment was attempted.",
            "guard": None,
            "checkout": None,
        }

    # "I've paid" / "confirm my payment" -> ask Razorpay for the real status.
    # Deterministic: we only report what the API returned, never a guess.
    if _wants_payment_status(message):
        audit.log(
            "query_received",
            session_id=session_id,
            data={"query": message, "intent": "payment_status_check"},
        )
        last_checkout = agent.last_checkout
        if last_checkout is None:
            return {
                "session_id": session_id,
                "reply": (
                    "I haven't created a payment link in this session yet, so "
                    "there's nothing to confirm. Search for a product first, "
                    "then tell me you want to buy it."
                ),
                "guard": None,
                "checkout": None,
            }
        try:
            pl = fetch_payment_link(last_checkout["link_id"])
        except Exception as exc:  # noqa: BLE001 - surface real API state
            audit.log(
                "razorpay_error",
                session_id=session_id,
                data={
                    "action": "payment_status_check",
                    "link_id": last_checkout["link_id"],
                    "error": str(exc)[:300],
                },
            )
            return {
                "session_id": session_id,
                "reply": (
                    "I tried to check with Razorpay but the lookup failed - I "
                    "won't guess whether it went through. Please check the "
                    "payment link in your test dashboard, or try again."
                ),
                "guard": None,
                "checkout": None,
            }
        status = (pl or {}).get("status", "unknown")
        paid = status == "paid"
        audit.log(
            "razorpay_response",
            session_id=session_id,
            data={
                "action": "payment_status_check",
                "link_id": last_checkout["link_id"],
                "status": status,
                "paid": paid,
            },
        )
        if paid:
            reset_cart = False
            if (
                agent.pending_checkout is not None
                and agent.pending_checkout.get("link_id") == last_checkout["link_id"]
            ):
                agent.cart = []
                agent.pending_checkout = None
                reset_cart = True
            reply = (
                f"Confirmed with Razorpay (TEST mode): your payment for "
                f"{last_checkout['name']} (Rs.{last_checkout['price']}) is "
                "complete - the link status is 'paid'."
            )
            if reset_cart:
                reply += " Your cart has been cleared."
            reply += " Thanks!"
        else:
            reply = (
                f"Razorpay reports that payment link "
                f"{last_checkout['link_id']} is not paid yet (status: "
                f"'{status}'). No money has moved in TEST mode. "
                f"Complete it here: {last_checkout['short_url']}"
            )
        return {
            "session_id": session_id,
            "reply": reply,
            "guard": None,
            "checkout": {
                "action": "payment_status_check",
                "link_id": last_checkout["link_id"],
                "status": status,
                "paid": paid,
            },
        }

    # Simulated payment decline (failure case c) - deterministic demo hook.
    if _wants_decline_simulation(message):
        audit.log(
            "query_received",
            session_id=session_id,
            data={"query": message, "intent": "decline_simulation"},
        )
        return _handle_decline_simulation(agent, session_id)

    # Normal path: remember which product the user is talking about, then run
    # the grounded agent (logs query_received, retrieval, guard).
    user_named = _find_named_in_results(
        message, (agent.last_search_results or {}).get("results") or []
    )
    if user_named is not None:
        agent.selected_product = user_named
    try:
        out = agent.run(message)
    except Exception as exc:  # noqa: BLE001 - demo-friendly degradation
        audit.log(
            "agent_error",
            session_id=session_id,
            data={"query": message, "error": str(exc)[:300]},
        )
        return {
            "session_id": session_id,
            "reply": (
                "My model service hiccuped (overloaded or quota-limited for a "
                "moment). Nothing was bought or charged - please try that again "
                "in a moment."
            ),
            "guard": None,
            "checkout": None,
        }
    if user_named is None:
        # User did not name a product -> infer it from what the agent said.
        _update_selection_from_reply(agent, out["final_text"])
    return {
        "session_id": session_id,
        "reply": out["final_text"],
        "guard": out.get("guard"),
        "checkout": None,
    }


@app.get("/api/logs")
def logs(session_id: str | None = None, limit: int = 50) -> dict[str, Any]:
    entries = audit.read()
    if session_id:
        entries = [e for e in entries if e.get("session_id") == session_id]
    return {"entries": entries[-limit:]}


# Serve the frontend last so /api routes take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
