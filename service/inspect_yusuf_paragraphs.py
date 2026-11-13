from sqlalchemy import select
from app.db import SessionLocal
from app.models import Paragraph, ParagraphTranslation

with SessionLocal() as session:
    rows = session.execute(
        select(
            Paragraph.id,
            Paragraph.position,
            ParagraphTranslation.lang,
            ParagraphTranslation.status,
            ParagraphTranslation.body,
        )
        .join(
            ParagraphTranslation,
            ParagraphTranslation.paragraph_id == Paragraph.id,
        )
        .where(Paragraph.story_id == "yusuf")
        .order_by(Paragraph.position)
    ).all()

    for row in rows:
        body = row.body[:120] if row.body else None
        print(
            f"{row.id} | pos={row.position} | lang={row.lang} "
            f"| status={row.status} | body={body!r}"
        )
