from app.db import SessionLocal
from app.models import ParagraphTranslation
from app.ml.kg_bridge import run_kg_extraction

PARAGRAPH_ID = "yusuf-p2"
LANG = "ar"

with SessionLocal() as session:
    row = session.get(
        ParagraphTranslation,
        {
            "paragraph_id": PARAGRAPH_ID,
            "lang": LANG,
        },
    )

    if row is None:
        raise SystemExit("Paragraph not found")

    result = run_kg_extraction(
        story_id="yusuf",
        story_key="yusuf",
        paragraph_id=PARAGRAPH_ID,
        lang=LANG,
        text=row.body,
    )

    print("TEXT:", repr(row.body))
    print("\nDIAGNOSTICS:")
    print(result.diagnostics)

    print("\nENTITIES:")
    for x in result.records.entities:
        print(x)

    print("\nMENTIONS:")
    for x in result.records.mentions:
        print(x)

    print("\nRELATIONS:")
    for x in result.records.relations:
        print(x)

    print("\nSTORY RELATIONS:")
    for x in result.records.story_relations:
        print(x)
