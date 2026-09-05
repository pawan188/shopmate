"""
Deterministic grounding validator: allows a purchase only when the claimed product id and price exactly match what retrieval returned.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from catalog.catalog import get_product_by_id  # noqa: E402


def _coerce_price(price: Any) -> int | None:
    """Accept int or digit-string prices ('3999'); anything else is not a price."""
    if isinstance(price, bool):  # True/False are ints in Python; never a price
        return None
    if isinstance(price, int):
        return price
    if isinstance(price, str) and price.strip().isdigit():
        return int(price.strip())
    return None


def validate_purchase(
    product_id: str,
    price: Any,
    last_search_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether (product_id, price) may proceed to checkout.

    Args:
        product_id: the id the flow wants to charge for.
        price: the amount the flow is about to charge (int or digit string).
        last_search_results: the session's most recent search_catalog() return
            value (or None if no search has happened this session).

    Returns:
        {"decision": "ALLOW" | "BLOCK", "product_id", "claimed_price",
         "retrieved_price", "catalog_price", "failures": [..], "grounded": bool}
    """
    failures: list[str] = []

    claimed = _coerce_price(price)
    if claimed is None:
        failures.append(f"price_not_parseable:{price!r}")

    # 1. A retrieval must have happened this session.
    results = (last_search_results or {}).get("results") or []
    if not results:
        failures.append("no_search_result_this_session")

    # 2. The claimed product must be among what was retrieved.
    retrieved_row: dict[str, Any] | None = None
    for row in results:
        if row.get("product_id") == product_id:
            retrieved_row = row
            break
    if retrieved_row is None:
        failures.append("product_id_not_in_retrieved_results")

    # 3. The claimed price must EXACTLY match the retrieved price.
    retrieved_price: int | None = retrieved_row.get("price") if retrieved_row else None
    if retrieved_row is not None:
        retrieved_price = _coerce_price(retrieved_row.get("price"))
        if claimed is not None and claimed != retrieved_price:
            failures.append(f"price_mismatch:claimed={claimed},retrieved={retrieved_price}")

    # 4. Belt-and-braces against the live catalog (never trust a stale claim).
    catalog_product = get_product_by_id(product_id) if product_id else None
    catalog_price: int | None = None
    if catalog_product is None:
        failures.append("product_id_not_in_live_catalog")
    else:
        catalog_price = _coerce_price(catalog_product["price"])
        if claimed is not None and claimed != catalog_price:
            failures.append(
                f"price_mismatch_vs_live_catalog:claimed={claimed},actual={catalog_price}"
            )
        if catalog_product.get("stock", 0) <= 0:
            failures.append("out_of_stock")

    allowed = not failures
    return {
        "decision": "ALLOW" if allowed else "BLOCK",
        "product_id": product_id,
        "claimed_price": claimed,
        "retrieved_price": retrieved_price,
        "catalog_price": catalog_price,
        "failures": failures,
        "grounded": allowed,
    }
