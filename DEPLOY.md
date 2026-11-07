# Deploying the NELP scoring service

I could not find RumptyCloud's documentation, so these steps are generic. Read its setup docs first
(hour-0 item in the proposal) and map each step below onto what it offers.

## What to deploy
Files: `Dockerfile`, `requirements-prod.txt`, `work_passport.py`, `llm_tools.py`,
`generate_synthetic_data.py`, `scoring_api.py` (all in `data_science/`).
Start command (what the Dockerfile runs): `uvicorn scoring_api:app --host 0.0.0.0 --port ${PORT}`
Health check path: `/health`   Port: the host's `PORT` variable, default 8000.

## 1. Make two secrets (run locally, keep the output private)
```
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Run it twice: one value is `API_KEY` (teammates use it), the other is `ADMIN_KEY` (only you and the demo page).

## 2. Set environment variables on the host (its secrets / env settings page, never in git)
`API_KEY`, `ADMIN_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
Optional: `CORS_ORIGINS` (only if a separate frontend in the browser calls this), `CLAUDE_MODEL`, `OPENAI_MODEL`.

## 3. Deploy by whichever route the docs allow
- **Container supported:** build from the `Dockerfile` (push an image to its registry, or connect the GitHub repo).
- **Python app supported without Docker:** build command `pip install -r requirements-prod.txt`,
  start command as above. Use Python 3.11 or newer.

## 4. Test the live URL (replace the URL and keys)
```
curl https://YOUR-URL/health
curl https://YOUR-URL/api/config -H "X-API-Key: YOUR_API_KEY"
curl -X POST https://YOUR-URL/api/attack-demo -H "X-API-Key: YOUR_API_KEY" \
     -H "X-Admin-Key: YOUR_ADMIN_KEY" -H "Content-Type: application/json" -d '{"level":"naive"}'
```
Expected: `{"status":"ok"}`, the config JSON, then a fake rider scoring about 16 (Low, status "flagged").
A call without the key must return 401. If it returns data, `API_KEY` is not set on the host.
Check that the attack demo says `"jobs_source": "openai"` and `"source": "claude"`; "offline-stub" and
"template" mean the keys were not picked up.

## 5. Give teammates
The base URL, the `API_KEY`, and `API.md`. Interactive docs are at `https://YOUR-URL/docs`.
Never send `ADMIN_KEY` to anyone who does not need the attack demo.

## Things to check in RumptyCloud's docs
- Does it run Docker images, or only a git repo plus a start command?
- Which port does it expect, and does it set `PORT`?
- Where do environment variables or secrets go?
- Does it sleep an idle app (a slow first request in the demo)?
- Is there a free-tier limit on memory (this service needs under 512 MB) or on outbound calls to OpenAI and Anthropic?
- Does it deploy automatically from GitHub (CI/CD)?

## Not tested
The image has not been built here (no Docker daemon in my environment). The start command was tested
in a clean Python environment with only `requirements-prod.txt` installed: health, genuine demo and
attack demo all worked.
