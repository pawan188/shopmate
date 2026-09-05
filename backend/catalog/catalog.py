"""
Catalog loader: reads products.json and exposes the authoritative product accessors used by retrieval and validation.
"""

import json
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).parent / "products.json"


def get_all_products() -> list[dict[str, Any]]:
    """Load and return all products from the catalog."""
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)


def get_product_by_id(product_id: str) -> dict[str, Any] | None:
    """Look up a single product by its ID. Returns None if not found."""
    products = get_all_products()
    for product in products:
        if product["id"] == product_id:
            return product
    return None


def get_products_by_category(category: str) -> list[dict[str, Any]]:
    """Return all products in a given category."""
    products = get_all_products()
    return [p for p in products if p["category"].lower() == category.lower()]
