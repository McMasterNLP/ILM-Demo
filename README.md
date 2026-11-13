# Ilm

An AI-powered Arabic learning platform for the Stories of the Prophets.

- `service/` — FastAPI with PostgreSQL 16 + pgvector ([backend details](https://github.com/McMasterNLP/ILM-Demo/blob/main/service/README.md))
- `frontend/` — React + Vite, with English and Arabic support

## Prerequisites

- Docker Desktop with Docker Compose
- Node.js and npm (for running the frontend locally)

## Configuration

Copy the example environment file to `service/.env`:

```powershell
Copy-Item service/.env.example service/.env
```

On macOS or Linux, use `cp service/.env.example service/.env` instead. Edit `service/.env` to configure the application. **Do not commit this file or share your API keys.**

### `COHERE_API_KEY`

Get an API key from [Cohere](https://dashboard.cohere.com/api-keys).

Cohere embeds approved paragraphs using `embed-multilingual-v3.0` (English and Arabic in a shared space) and reranks search results.

**If blank:** paragraphs cannot be indexed, publishing a story is refused, and `/search` returns HTTP 503. You can still explore already-seeded development content.

### `GEMINI_API_KEY`

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

Gemini generates quiz questions from approved paragraphs and grades written answers against retrieved passages. It does not write story text.

**If blank:** other features remain available, but question generation returns HTTP 503.

Free-tier limits may change; check your account's current quotas before a demo. Generating six questions can require six model calls.

### Admin credentials

Set `ADMIN_PASSWORD` and `SESSION_SECRET` in `service/.env` to access the authoring console. See `service/.env.example` for other settings.

## Run the backend

From the repository root:

```powershell
docker compose up --build -d
```

This starts PostgreSQL and the FastAPI service and runs database migrations. The API is available at http://localhost:8000.

## Load development stories (optional)

For a local demo, seed the database:

```powershell
docker compose exec service python -m app.seed
```

The seed script populates six stories: Yusuf, Nuh, Musa, Yunus, Ibrahim, and Salih. This is **development content**, not a complete curated collection: Yusuf includes some text and questions, while other stories and some paragraphs are placeholders. Do not treat the seeded content as publication-ready.

## Run the frontend

In a second terminal, from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The frontend proxies `/api` requests to the backend, so start the backend first.

## Useful URLs

| URL | Purpose |
| --- | --- |
| http://localhost:5173 | Learner-facing app |
| http://localhost:8000/admin/authoring | Authoring console: upload, approve, and generate questions |
| http://localhost:8000/api/v1/docs | Interactive API documentation |

## Publish your first story

1. Sign in at http://localhost:8000/admin/authoring.
2. Paste a story. **Blank lines separate paragraphs**; each paragraph becomes one retrievable chunk.
3. Select **Save**. Paragraphs are stored as approved and indexed automatically when embedding is configured.
4. Select **Publish**. Publication is refused until all paragraphs are indexed.
5. Select **Generate questions**. Review each question alongside the paragraph it came from before approving it.

Published stories appear in the learner-facing app. The quiz unlocks after the learner finishes reading.

## Run backend tests

Tests should use a separate database, not the application database. Create it once if it does not already exist:

```powershell
docker compose exec db psql -U ilm -d ilm -c "CREATE DATABASE ilm_test;"
```

From the `service/` directory in **PowerShell**, run:

```powershell
docker compose -f ..\docker-compose.yml run --rm --no-deps -v "${PWD}:/app" -e "TEST_DATABASE_URL=postgresql+psycopg://ilm:ilm@db:5432/ilm_test" service python -m pytest -q
```

At the time of local verification, all 21 backend tests passed. The test command mounts the local `service/` directory because the test files are not included in the service image.

## Notes for contributors

- Keep `.env` and API keys out of Git.
- The development seed data contains placeholders; replace them with reviewed material before publishing real content.
- The existing Alembic migration chain includes a legacy `Base.metadata.create_all()` call in `0001_initial.py`. The later migration handles an already-existing `attempt_questions` table, but clean-database migrations and downgrade behavior should be reviewed before production deployment.

Additional settings are documented in `service/.env.example`.
