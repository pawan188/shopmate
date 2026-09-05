"""
Deterministic low-confidence guard: product/price claims without solid retrieval backing are replaced by a clarifying question.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from catalog.catalog import get_all_products  # noqa: E402

CATALOG_PRICES = {p["price"] for p in get_all_products()}

# "Rs. 3999", "INR 3999", "rupees 3999", "Rs3999", or a rupee-sign-prefixed number.
CURRENCY_RE = re.compile(
    r"(?:inr|rs\.?|rupees?)\s*([\d,]+)|\u20b9\s*([\d,]+)", re.IGNORECASE
)

CLARIFY_QUESTION = (
    "I'm not confident enough about which product you mean to make a "
    "recommendation yet. Could you narrow it down a little - for example, "
    "what category you're looking for, what you'll use it for, or a budget range?"
)


def mentions_price(text: str) -> bool:
    """True if the text states a price: a currency token, or a bare number that
    exactly equals a catalog price (the agent writing '3999' with no symbol)."""
    for match in CURRENCY_RE.finditer(text):
        for group in match.groups():
            if group:
                return True
    for token in re.findall(r"\b\d{3,6}\b", text):
        if int(token) in CATALOG_PRICES:
            return True
    return False


def mentions_retrieved_product(text: str, last_search_results: dict[str, Any] | None) -> bool:
    """True if the text names any product (name or id) from the last retrieval."""
    if not last_search_results:
        return False
    lowered = text.lower()
    for row in last_search_results.get("results", []):
        name = (row.get("name") or "").lower()
        if name and name in lowered:
            return True
        product_id = (row.get("product_id") or "").lower()
        if product_id and product_id in lowered:
            return True
    return False


def decide_response(
    final_text: str,
    last_search_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply the low-confidence guard to a model answer.

    Returns {"override": bool, "reason": str, "text": str} where text is what
    the user should actually see.
    """
    low_confidence = bool(
        last_search_results and last_search_results.get("low_confidence")
    )
    if not low_confidence:
        return {
            "override": False,
            "reason": "confidence_ok",
            "text": final_text,
        }

    if mentions_price(final_text) or mentions_retrieved_product(
        final_text, last_search_results
    ):
        return {
            "override": True,
            "reason": "low_confidence_product_claim",
            "text": CLARIFY_QUESTION,
        }

    return {
        "override": False,
        "reason": "low_confidence_already_clarifying",
        "text": final_text,
    }
