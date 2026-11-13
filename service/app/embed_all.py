from __future__ import annotations

import argparse

from app.config import settings
from app.db import SessionLocal
from app.embeddings import EmbeddingUnavailable
from app.repositories import authoring as authoring_repo
from app.repositories import retrieval as retrieval_repo

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks that have no vector yet.")
    parser.add_argument(
        "--batch", type=int, default=200, help="chunks per pass (default 200)"
    )
    args = parser.parse_args()

    model = settings.embedding_model
    total = 0

    with SessionLocal() as session:
        run = authoring_repo.start_run(session, stage="embedding", model=model)
        try:
            while True:
                chunks = retrieval_repo.unembedded_chunks(session, model, limit=args.batch)
                if not chunks:
                    break
                total += retrieval_repo.embed_chunks(session, chunks, run=run)
                session.commit()
                print(f"  embedded {total}…")
        except EmbeddingUnavailable as exc:
            session.rollback()
            raise SystemExit(f"Embedding unavailable: {exc}") from exc

        authoring_repo.finish_run(session, run)
        session.commit()

    print(f"Done. {total} chunks embedded with {model}.")

if __name__ == "__main__":
    main()
