from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, Source

class UnsupportedDocument(ValueError):
    pass

def extract_text(path: Path, media_type: str | None) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf" or (media_type or "").endswith("pdf"):
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if suffix in {".txt", ".md", ".markdown"} or (media_type or "").startswith("text/"):
        return path.read_text(encoding="utf-8", errors="replace")

    raise UnsupportedDocument(f"Cannot read {path.name} — upload a PDF, TXT or MD file")

def store_upload(filename: str, data: bytes, document_id: str) -> Path:
    directory = Path(settings.upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{document_id}{Path(filename).suffix.lower() or '.txt'}"
    target.write_bytes(data)
    return target

def get_or_create_source(session: Session, title: str) -> Source:
    existing = session.scalars(select(Source).where(Source.title == title)).first()
    if existing:
        return existing
    source = Source(title=title)
    session.add(source)
    session.flush()
    return source

def document_chunks(session: Session, document_id: str) -> list[Chunk]:
    stmt = select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.position)
    return list(session.scalars(stmt))
