"""
Embeds catalog products with a local sentence-transformer and stores the vectors in a local Chroma collection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the sibling `catalog` package importable whether this file is run as a
# script (python backend/verify_step2.py) or imported as a module.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from catalog.catalog import get_all_products  # noqa: E402

import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "electronics_catalog"
CHROMA_PATH = _BACKEND_DIR / "chroma_db"

# Lazily-initialised singletons: load the model / open the DB once, then reuse.
_model: SentenceTransformer | None = None
_client: Any = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once (lazy singleton).

    First call downloads ~90 MB from HuggingFace into the local HF cache;
    every call after that reuses the cached weights (no network).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_client() -> Any:
    """Persistent Chroma client (vectors live on disk under backend/chroma_db)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def product_to_text(product: dict[str, Any]) -> str:
    """The text representation we embed (and store as the document) per product.

    We fold in name + description + category + variants + price so a query
    phrased by use-case, category, or attribute (e.g. a colour) can all find
    the right item. The discriminating signal is really name + description;
    the rest is cheap extra recall.
    """
    variants = ", ".join(product.get("variants", []))
    return (
        f"{product['name']}. {product['description']} "
        f"Category: {product['category']}. "
        f"Variants: {variants}. "
        f"Price: {product['price']} INR."
    )


def build_collection(force: bool = False) -> Any:
    """Embed every catalog product and (re)store it in Chroma. Returns the collection.

    Idempotent: if the collection already holds exactly the current number of
    products, we reuse it as-is (fast startup, no re-embedding). Pass force=True
    to rebuild from scratch -- do this whenever the catalog changes, otherwise
    the vectors go stale and retrieval would ground on old data.
    """
    client = _get_client()
    products = get_all_products()

    try:
        existing = client.get_collection(COLLECTION_NAME)
    except Exception:
        existing = None

    # Already built and in sync -> reuse.
    if existing is not None and not force and existing.count() == len(products):
        return existing

    # Otherwise rebuild cleanly so we never leave stale/duplicate vectors behind.
    if existing is not None:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    model = _get_model()
    texts = [product_to_text(p) for p in products]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=[p["id"] for p in products],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "name": p["name"],
                "price": p["price"],
                "stock": p["stock"],
                "category": p["category"],
            }
            for p in products
        ],
    )
    return collection


def search(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    """Embed `query` and return the top-N nearest catalog products.

    Each result is a dict: id, name, price, stock, category, confidence, distance.
      confidence = 1 - cosine_distance, clamped to [0, 1] -- higher is a closer
      semantic match. This is the number Step 6 thresholds on.
    """
    collection = build_collection()  # ensures the store exists / is in sync
    model = _get_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    res = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    results: list[dict[str, Any]] = []
    for pid, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0]):
        confidence = max(0.0, 1.0 - float(dist))
        results.append(
            {
                "id": pid,
                "name": meta["name"],
                "price": meta["price"],
                "stock": meta["stock"],
                "category": meta["category"],
                "confidence": round(confidence, 4),
                "distance": round(float(dist), 4),
            }
        )
    return results


if __name__ == "__main__":
    # `python backend/catalog/embeddings.py [--force]` -> (re)build the store.
    collection = build_collection(force="--force" in sys.argv)
    print(f"Built '{COLLECTION_NAME}' with {collection.count()} products at {CHROMA_PATH}")
