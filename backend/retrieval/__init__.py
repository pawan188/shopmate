"""
Retrieval layer: the agent's single tool (see search.py).
"""

from .search import CONFIDENCE_THRESHOLD, search_catalog

__all__ = ["search_catalog", "CONFIDENCE_THRESHOLD"]
