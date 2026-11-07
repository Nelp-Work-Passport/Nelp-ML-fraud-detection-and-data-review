# NELP scoring service: how to connect

A stateless FastAPI service. Send jobs in, get a score, fraud flags and an explanation out.
It has no database, SMS or login. The site owns those.

## Run
```
pip install -r requirements.txt
uvicorn scoring_api:app --host 0.0.0.0 --port 8000
```
Live, clickable docs: `http://localhost:8000/docs`.
To mount it inside the main site instead: `from scoring_api import router` then `app.include_router(router)`.

## Environment variables (host secrets, never in git)
| Name | Purpose |
|---|---|
| `API_KEY` | if set, every `/api/*` call needs header `X-API-Key` |
| `ADMIN_KEY` | if set, `/api/attack-demo` also needs header `X-Admin-Key` |
| `OPENAI_API_KEY` | real LLM-written fake riders (offline stub if missing) |
| `ANTHROPIC_API_KEY` | Claude explanations (deterministic template if missing) |
| `CORS_ORIGINS` | only if a separate frontend calls this from the browser |

If `API_KEY` / `ADMIN_KEY` are unset the endpoints are open. Set both before deploying.

## Endpoints
| Method and path | Who calls it | Purpose |
|---|---|---|
| `GET /health` | host, uptime check | liveness |
| `POST /api/score` | profile page, after each confirmation | score one rider; send `peers` for ring and shared-device checks; `explain: true` adds the text |
| `POST /api/score/batch` | recompute job | many riders at once with full cross-rider checks |
| `POST /api/explain` | profile page | explanation for an `assessment` you already have |
| `GET /api/config` | profile page | weights, caps, bands and thresholds to display |
| `POST /api/attack-demo` | admin demo page | OpenAI fabricates a rider, we score it low and explain why |
| `GET /api/demo/genuine` | demo page | a seeded genuine rider that scores high |

## Example
```
curl -X POST http://localhost:8000/api/score -H "Content-Type: application/json" -d '{
  "rider": {
    "rider_id": "amina-k", "phone": "0712345678", "signup_at": "2026-09-01T08:00:00Z",
    "jobs": [
      {"customer_id": "0722000001", "logged_at": "2026-09-03T09:10:00Z", "status": "CONFIRMED",
       "confirmed_at": "2026-09-03T09:25:00Z", "confirm_device": "dev_abc", "rider_device": "rdev_amina"}
    ]
  },
  "explain": true, "lang": "en"
}'
```
Response: `{"assessment": {total, band, status, label, components[], flags[], facts}, "explanation": {text, source}}`

## Rules for callers
- Phone numbers are text in ONE normalised format (for example `2547XXXXXXXX`). Never numbers: leading zeros are lost.
- Send every job the rider has, with its real `status`. Only `CONFIRMED` jobs count; the rest are ignored.
- `total` is `null` and `band` is "Not enough evidence yet" below 5 confirmed jobs. Show that text, not a score.
- `status` is `ok`, `caution` (medium flag, -15 points), `flagged` (high flag, capped at 20) or `insufficient`.
- Show `flags[].reason` on the profile as is. It is plain language and already contains the numbers.
- Never show full customer phone numbers publicly.
- Timestamps: ISO 8601, ideally UTC.
