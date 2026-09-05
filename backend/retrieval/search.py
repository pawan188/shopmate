"""
Catalog search: embeds a query, runs local vector search, and joins hits to live catalog records with confidence scores. This is the agent's only tool.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the sibling `catalog` package importable whether we're imported as a
# module or a verify script is run directly.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from catalog.catalog import get_product_by_id  # noqa: E402
from catalog.embeddings import search as vector_search  # noqa: E402

# Below this top-1 confidence the agent must ask a clarifying question instead of
# assuming which product the user meant. Enforced in Step 6; defined here because
# the retrieval layer is what produces the confidence being thresholded.
#
# Calibrated from observed scores, NOT a round guess: all-MiniLM-L6-v2 yields
# cosine similarities of ~0.54-0.57 for clearly-correct matches and ~0.32-0.39 for
# genuinely ambiguous queries on this catalog, so 0.45 sits in the gap. (The
# handover's 0.6 would flag even perfect matches as "unsure".) Finalized in Step 6.
CONFIDENCE_THRESHOLD = 0.45


def search_catalog(query: str, n_results: int = 5) -> dict[str, Any]:
    """Search the catalog by natural-language query. This is the agent's only tool.

    Returns:
        {
          "query": <str>,
          "results": [                       # sorted best-first
            {"product_id", "name", "price", "stock", "category",
             "description", "variants", "confidence"},
            ...
          ],
          "top_confidence": <float>,          # confidence of best match, 0.0 if none
          "low_confidence": <bool>,           # top_confidence < CONFIDENCE_THRESHOLD
        }

    price / stock / description / variants are read from the LIVE catalog record
    (not the vector store), so they are always authoritative and safe to ground on.
    """
    hits = vector_search(query, n_results=n_results)

    results: list[dict[str, Any]] = []
    for hit in hits:
        product = get_product_by_id(hit["id"])
        if product is None:
            # Vector store referenced an id that's no longer in the catalog.
            # Skip it rather than surface a product we can't ground.
            continue
        results.append(
            {
                "product_id": product["id"],
                "name": product["name"],
                "price": product["price"],
                "stock": product["stock"],
                "category": product["category"],
                "description": product["description"],
                "variants": product["variants"],
                "confidence": hit["confidence"],
            }
        )

    top_confidence = results[0]["confidence"] if results else 0.0
    return {
        "query": query,
        "results": results,
        "top_confidence": top_confidence,
        "low_confidence": top_confidence < CONFIDENCE_THRESHOLD,
    }
