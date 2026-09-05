"""
Catalog loader: JSON-based product access.
"""

import json
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).parent / "products.json"


def load_catalog() -> list[dict[str, Any]]:
    """Load the full product catalog from the JSON file."""
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)


def get_product_by_id(catalog: list[dict], product_id: str) -> dict | None:
    """Look up a single product by its ID."""
    for product in catalog:
        if product["id"] == product_id:
            return product
    return None
