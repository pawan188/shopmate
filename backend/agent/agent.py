"""
Gemini agent loop. The model has exactly one tool (search_catalog); the loop executes tool calls, feeds results back, and applies the grounding guard before any answer is shown.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_BACKEND_DIR / ".env")

from google import genai  # noqa: E402
from google.genai import errors  # noqa: E402
from google.genai import types  # noqa: E402

from retrieval.search import search_catalog  # noqa: E402

from agent.clarify import decide_response  # noqa: E402

DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """\
You are a shopping assistant for an electronics store. Help the user find
products and answer questions about the catalog.

GROUNDING CONTRACT -- absolute, non-negotiable rules:
1. You have exactly ONE tool: search_catalog. It queries the store's real
   catalog and returns products with their exact prices (in INR), stock,
   descriptions and confidence scores.
2. You MUST call search_catalog before stating ANY product name, price, stock
   status, or availability. You are FORBIDDEN from answering product questions
   from memory, from guessing, and from making up prices or stock figures.
3. Only relay information the tool actually returned in this conversation.
   Never paraphrase or "correct" a price. If the tool returned
   low_confidence: true, you are NOT sure which product the user means -- ask a
   clarifying question instead of picking one.
4. If the user asks you to state a price or product directly without searching,
   politely refuse and call search_catalog first.
5. Be concise and friendly. Prices are in Indian Rupees (INR).
"""


def build_tool() -> types.Tool:
    """The single tool exposed to the model: search_catalog."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_catalog",
                description=(
                    "Search the store's real product catalog by a natural-language query. "
                    "Returns the top matching products with product_id, name, price (INR), "
                    "stock, category, description, variants and a confidence score (0-1). "
                    "Call this BEFORE stating any product, price, or stock information. "
                    "If the result has low_confidence=true, do not pick a product -- ask the "
                    "user a clarifying question instead."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="The shopper's request, phrased as a search query.",
                        ),
                        "n_results": types.Schema(
                            type=types.Type.INTEGER,
                            description="How many top results to return (1-5). Default 5.",
                            default=5,
                        ),
                    },
                    required=["query"],
                ),
            )
        ]
    )


def get_model_id() -> str:
    """Model id: GEMINI_MODEL env var if set, else the default Flash-Lite id.

    Model ids change as Google ships new generations. If the default is stale
    for your key, set GEMINI_MODEL in backend/.env. verify_step4.py lists the
    live model ids on your key so you can confirm.
    """
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


class ShoppingAgent:
    """One agent = one shopping session (keeps history + last search results).

    ``last_search_results`` is where Step 5's deterministic validator will read
    what the agent is allowed to act on. It is overwritten on every tool call.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        audit: Any | None = None,
        session_id: str | None = None,
    ) -> None:
        """audit: optional AuditLogger; session_id: stable id for log lines."""
        self.model = model or get_model_id()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Add it to backend/.env "
                "(GEMINI_API_KEY=...) or pass api_key=."
            )
        self.audit = audit
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.client = genai.Client(api_key=key)
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[build_tool()],
            temperature=0.2,  # low -> fewer surprise phrasings on stage
        )
        self.history: list[types.Content] = []
        self.last_search_results: dict[str, Any] | None = None
        self.last_checkout: dict[str, Any] | None = None
        self.selected_product: dict[str, Any] | None = None
        self.cart: list[dict[str, Any]] = []
        self.pending_checkout: dict[str, Any] | None = None

    def _run_tool(self, call: types.FunctionCall) -> dict[str, Any]:
        """Execute ONE tool call. Only search_catalog exists; anything else is blocked."""
        if call.name != "search_catalog":
            raise RuntimeError(
                f"Agent attempted to call disallowed tool '{call.name}' - blocked by the loop."
            )
        args = call.args or {}
        query = str(args.get("query", ""))
        n_results = int(args.get("n_results", 5))
        result = search_catalog(query, n_results=n_results)
        self.last_search_results = result
        return result

    def _generate_with_retry(self, retries: int = 3) -> Any:
        """Call Gemini, retrying transient overload/quota errors a few times.

        Gemini sometimes returns 429/503 under load; a short backoff usually
        gets through. Real auth/model errors still raise immediately.
        """
        for attempt in range(retries):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=self.history, config=self.config
                )
            except (errors.ServerError, errors.ClientError) as exc:
                code = getattr(exc, "code", None)
                if not (isinstance(exc, errors.ServerError) or code == 429):
                    raise
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("Gemini call failed after retries")  # pragma: no cover

    def run(self, user_message: str) -> dict[str, Any]:
        """One user turn, looped until the model answers with text (not a tool call).

        Returns {"final_text", "tool_calls", "last_search_results", "transcript",
                 "guard"} where guard = {"overridden", "reason"} describes the
        Step 6 low-confidence check on this turn's answer.
        """
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )
        if self.audit is not None:
            self.audit.log(
                "query_received",
                session_id=self.session_id,
                data={"query": user_message},
            )
        tool_calls: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = self._generate_with_retry()

            # No candidates -> surface the block reason instead of guessing.
            if not response.candidates:
                raise RuntimeError(
                    f"Gemini returned no candidates: {response.prompt_feedback}"
                )

            parts = response.candidates[0].content.parts
            has_tool_call = any(p.function_call is not None for p in parts)

            if not has_tool_call:
                raw_text = "".join(p.text or "" for p in parts)
                guard = decide_response(raw_text, self.last_search_results)
                final_text = guard["text"]
                if self.audit is not None:
                    self.audit.log(
                        "clarify_guard",
                        session_id=self.session_id,
                        data={
                            "overridden": guard["override"],
                            "reason": guard["reason"],
                            "final_text": final_text[:500],
                        },
                    )
                # Keep history truthful: store exactly what the user sees.
                self.history.append(
                    types.Content(role="model", parts=[types.Part(text=final_text)])
                )
                return {
                    "final_text": final_text,
                    "tool_calls": tool_calls,
                    "last_search_results": self.last_search_results,
                    "transcript": self._transcript(),
                    "guard": {
                        "overridden": guard["override"],
                        "reason": guard["reason"],
                    },
                }

            # Tool-call turn: execute each call, then feed the results back.
            self.history.append(response.candidates[0].content)
            tool_result_parts: list[types.Part] = []
            for part in parts:
                call = part.function_call
                if call is None:
                    continue
                result = self._run_tool(call)
                tool_calls.append(
                    {
                        "tool": call.name,
                        "query": (call.args or {}).get("query"),
                        "result": result,
                    }
                )
                if self.audit is not None:
                    self.audit.log(
                        "retrieval_results",
                        session_id=self.session_id,
                        data={
                            "tool": "search_catalog",
                            "query": result.get("query"),
                            "top_confidence": result.get("top_confidence"),
                            "low_confidence": result.get("low_confidence"),
                            "results": result.get("results"),
                        },
                    )
                tool_result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call.id, name=call.name, response=result
                        )
                    )
                )
            # Gemini expects tool results as a user-role content; the id ties
            # each result to the call that requested it.
            self.history.append(types.Content(role="user", parts=tool_result_parts))

        raise RuntimeError(
            f"Agent exceeded {MAX_TOOL_ROUNDS} tool rounds without answering - "
            "stopping to avoid an infinite loop."
        )

    def _transcript(self) -> list[dict[str, Any]]:
        """Human-readable transcript of this session's history, for logging/verification."""
        out: list[dict[str, Any]] = []
        for content in self.history:
            for part in content.parts or []:
                if part.function_call is not None:
                    out.append(
                        {
                            "role": content.role,
                            "type": "function_call",
                            "tool": part.function_call.name,
                            "args": part.function_call.args,
                        }
                    )
                elif part.function_response is not None:
                    out.append(
                        {
                            "role": content.role,
                            "type": "function_response",
                            "tool": part.function_response.name,
                        }
                    )
                elif part.text:
                    out.append({"role": content.role, "type": "text", "text": part.text})
        return out
