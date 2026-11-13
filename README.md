# Ilm

An AI-powered Arabic learning platform for the Stories of the Prophets.

- `service/` — FastAPI on Postgres 16 + pgvector ([details](service/README.md))
- `frontend/` — React + Vite, English and Arabic

## Keys

```bash
cp service/.env.example service/.env
```

Two keys go in `service/.env`.

### `COHERE_API_KEY`

Free key at [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).

Embeds every approved paragraph with `embed-multilingual-v3.0` — English and
Arabic in one shared space — and reranks search results.

**Blank:** paragraphs are never indexed, publishing a story is refused, and
`/search` returns 503.

### `GEMINI_API_KEY`

Free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Writes quiz questions from one approved paragraph at a time, and grades written
answers against retrieved passages. It never writes story text.

**Blank:** everything else works; question generation returns 503.

> The free tier allows **20 requests per day**. One question costs one call, so
> six questions costs six. Use a billed key before demoing.

The same file also holds `ADMIN_PASSWORD` and `SESSION_SECRET` — set both, or
the authoring console will not let you in.

## Run the backend

```bash
docker compose up --build -d
```

Postgres and the API, migrations included. Serves on http://localhost:8000.

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

http://localhost:5173 — it proxies `/api` to the backend, so start the backend
first.

| | |
|---|---|
| http://localhost:5173 | the app |
| http://localhost:8000/admin/authoring | upload stories, approve, generate questions |
| http://localhost:8000/api/v1/docs | API |

## First story

1. Sign in at `/admin/authoring`
2. Paste a story — **blank lines separate paragraphs**, and one paragraph
   becomes one retrievable chunk
3. **Save** — paragraphs are stored approved and indexed automatically
4. **Publish** — refused until every paragraph is indexed
5. **Generate questions** — each is shown beside the paragraph it came from,
   so approving is a comparison rather than a judgement

It appears at http://localhost:5173 once published. The quiz unlocks after a
learner finishes reading.

Every other setting is documented in `service/.env.example`.
