# ShopMate — Grounded Agentic Commerce

A hackathon project: an AI shopping agent that lets you shop in natural language, where **hallucinating a product, price, or stock figure is structurally impossible** — and completed purchases run on **Razorpay TEST mode** keys only.

The agent can state a product/price/stock figure only when it has just retrieved it from a real catalog in that same conversation. A deterministic validator (plain code, not another LLM call) blocks any checkout whose product id or price does not exactly match what retrieval returned. Everything is written to an append-only audit log.

## Why it can't hallucinate

- **One tool.** The agent (Gemini) has exactly one tool: `search_catalog`. The loop refuses to execute anything else.
- **Grounded answers.** Prices, stock, and descriptions are only relayed from the tool's output — never from the model's memory.
- **Deterministic validator.** Checkout (single product or full cart) is allowed only when the claimed `product_id` + `price` exactly match what retrieval returned; cart line items are re-verified against the live catalog at payment time.
- **Clarify guard.** If the top retrieval match falls below the confidence threshold (0.45), a deterministic guard replaces any would-be product/price claim with a clarifying question — the agent never guesses which product you mean.
- **TEST-only payments.** The Razorpay client refuses to run with anything that isn't an `rzp_test_` key.
- **Audit trail.** Query → retrieval + scores → validator decision → Razorpay response are all logged to append-only JSON Lines.

## Tech stack

| Layer | What |
| --- | --- |
| Backend | Python + FastAPI (`backend/main.py`) |
| Vector store | Chroma (local, persisted in `backend/chroma_db`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, no API key, no network at query time) |
| Agent | Gemini Flash Lite (`gemini-3.1-flash-lite`) with native function calling |
| Payments | Razorpay TEST mode — payment links + status lookup |
| Frontend | Single-file chat UI served by FastAPI (`frontend/index.html`) |

## Project layout

```
backend/
  main.py                 # FastAPI app: /api/chat, /api/logs, /health, static frontend
  agent/                  # Gemini loop, single search_catalog tool, clarify guard
  catalog/                # products.json, loader, embedding pipeline (Chroma)
  retrieval/              # search_catalog() with confidence scores
  validator/              # deterministic purchase validator
  razorpay/               # TEST-mode payment client (validator-gated)
  logging_utils/          # append-only JSON Lines audit logger
  audit_log/              # runtime audit log (gitignored)
  tests/                  # per-layer verification scripts
frontend/                 # chat UI
catalog/products.json     # seeded catalog (75 fake products) — single source of truth
```

## Setup

Prereqs: Python 3.11+.

```powershell
# 1. Create the venv and install dependencies (from the repo root)
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install fastapi "uvicorn[standard]" httpx python-dotenv chromadb sentence-transformers google-genai

# 2. Configure environment
cd backend
copy .env.example .env      # then fill in real values (see below)
```

`.env` keys (`backend/.env` is gitignored — never commit it):

| Variable | Required | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | From https://aistudio.google.com/apikey (free tier) |
| `GEMINI_MODEL` | No | Default `gemini-3.1-flash-lite` |
| `RAZORPAY_KEY_ID` | For checkout | TEST keys only, must start `rzp_test_` |
| `RAZORPAY_KEY_SECRET` | For checkout | TEST keys only |

```powershell
# 3. Build the vector store once (downloads the embedding model on first run,
#    ~90 MB; afterwards retrieval is fully local/offline)
python -m catalog.embeddings

# 4. Run the server
python -m uvicorn main:app --port 8000
```

Open http://127.0.0.1:8000 — the chat UI loads and talks to the backend.

## Using it

| You say | What happens |
| --- | --- |
| `wireless noise cancelling headphones` | Grounded list from catalog, with prices/stock |
| `add the airpods pro to my cart` / `add it` | Validator-gated add to cart (cart count shown in header) |
| `what's in my cart` | Itemized cart + total |
| `remove the nothing ear` / `clear my cart` | Remove line / empty cart |
| `checkout` or `checkout my cart` | One Razorpay TEST payment link for the validated sum; cart is kept until payment succeeds |
| `i have paid` / `can you check if the payment has been made` | Asks Razorpay for the real link status; on `paid` the cart clears |
| `simulate a payment decline` | Demo hook: logs a marked `payment_decline` (simulated), never auto-retries, keeps the cart |
| `something for my desk` | Ambiguous → clarifying question (deterministic guard) |
| `the sony is Rs. 24991, right?` | Agent only repeats the retrieved price (Rs. 24,990) |

To pay in TEST mode: open the payment link and choose **Netbanking → any bank**. (Card testing can show "International cards not supported" on some test accounts.)

## API

- `GET /health` → `{"status": "ok", "active_sessions": N}`
- `POST /api/chat` — body `{"message": "...", "session_id": "..."}` (session id optional; kept in browser localStorage). Response:
  ```json
  {
    "session_id": "abc123",
    "reply": "...",
    "guard": {"overridden": false, "reason": "confidence_ok"},
    "checkout": null,
    "cart": {"count": 0, "total_inr": 0, "items": []}
  }
  ```
  `guard` shows clarify-guard decisions; `checkout` carries payment-link or status-check details; `cart` is the session cart snapshot.
- `GET /api/logs?session_id=...&limit=50` → recent audit entries.

## Tests

Verification scripts live in `backend/tests/` (run from `backend/` with `..\.venv\Scripts\python.exe tests\verify_stepX.py`). Highlights:

- `verify_step5.py` — validator: exact match ALLOW; tampered price / unretrieved id / no-search BLOCK (offline).
- `verify_step6.py` — low-confidence queries end in a clarifying question, never a guess.
- `verify_step7.py` — TEST payment-link creation behind the validator gate (needs Razorpay test keys).
- `verify_step8.py` — append-only audit trail captures every stage.

## Demo & docs

- `DEMO_SCRIPTS.md` — 3 rehearsed demo scripts (happy path, clarify guard, tamper + decline) with expected outputs and judge talking points.
- `VIDEO_SCRIPT.md` — timed 5-minute video narration script.
- `HANDOVER.md` — full build log (each layer's what/why/verification).

## Limitations & honest notes

- The catalog is seeded with ~75 fake products; retrieval, grounding, validation, payment, and logging are real.
- Sessions and carts are in-memory: a server restart clears them.
- Razorpay payment links carry a single total amount; the per-item breakdown lives in the link notes, the chat reply, and the audit log.
- The simulated decline is a clearly-marked deterministic demo hook (logged `simulated: true`) — no Razorpay decline event is fabricated.
- All keys are TEST-mode; the payment client refuses non-`rzp_test_` keys so no real money can move.

**Built for the "AI Growth & Agentic Commerce" hackathon track.** No hallucinated prices — by construction.
