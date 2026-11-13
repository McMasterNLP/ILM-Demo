from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

from app.ml.kg_records import KGRecordBuilder
from app.ml.story_relation_pipeline import StoryRelationPipeline


_PIPELINE: StoryRelationPipeline | None = None


def _get_pipeline() -> StoryRelationPipeline:
    """
    Lazily create the heavyweight NLP pipeline once per Python process.

    This keeps the standalone runner compatible while allowing a future
    persistent HTTP service to reuse the loaded NER/RE models.
    """
    global _PIPELINE

    if _PIPELINE is None:
        _PIPELINE = StoryRelationPipeline()

    return _PIPELINE


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)

    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")

    if not value.strip():
        raise ValueError(f"{key} must not be empty")

    return value


def run_paragraph(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Run the complete KG extraction path for one approved Arabic paragraph.

    Input:
        story_id:
            Database Story.id used for StoryEntity / StoryRelation records.

        story_key:
            Canonical Islamic story key used by IslamicEntityLinker,
            e.g. "yusuf".

        paragraph_id:
            Database Paragraph.id used for EntityMention provenance and
            StoryRelation evidence.

        lang:
            Must currently be "ar". The NER/RE integration is Arabic-only.

        text:
            Approved ParagraphTranslation.body.

    This function:
        - performs no database access
        - performs no commit
        - does not modify the research NER/RE systems
        - runs entity linking only once
        - returns JSON-serializable logical KG records
    """
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")

    story_id = _require_string(payload, "story_id")
    story_key = _require_string(payload, "story_key")
    paragraph_id = _require_string(payload, "paragraph_id")
    lang = _require_string(payload, "lang")
    text = _require_string(payload, "text")

    if lang != "ar":
        raise ValueError(
            "KG extraction currently supports approved Arabic paragraphs only"
        )

    pipeline = _get_pipeline()

    result = pipeline.predict(
        text,
        story_key=story_key,
    )

    records = KGRecordBuilder().build(
        story_id=story_id,
        paragraph_id=paragraph_id,
        mentions=result.linked_mentions,
        relations=result.educational_relations,
    )

    return {
        "story_id": story_id,
        "story_key": story_key,
        "paragraph_id": paragraph_id,
        "lang": lang,

        "records": {
            "entities": [
                asdict(item)
                for item in records.entities
            ],
            "mentions": [
                asdict(item)
                for item in records.mentions
            ],
            "story_entities": [
                asdict(item)
                for item in records.story_entities
            ],
            "relations": [
                asdict(item)
                for item in records.relations
            ],
            "story_relations": [
                asdict(item)
                for item in records.story_relations
            ],
        },

        "diagnostics": {
            "linked_mentions": len(result.linked_mentions),
            "stage_a_candidates": len(result.stage_a_candidates),
            "stage_b_predictions": len(result.stage_b_predictions),
            "educational_relations": len(
                result.educational_relations
            ),
            "unresolved_source_relations": len(
                result.unresolved_source_relations
            ),
        },
    }


def main() -> None:
    try:
        payload = json.load(sys.stdin)

        result = run_paragraph(payload)

        json.dump(
            result,
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )

        sys.stdout.write("\n")

    except Exception as exc:
        error = {
            "error": type(exc).__name__,
            "message": str(exc),
        }

        json.dump(
            error,
            sys.stderr,
            ensure_ascii=False,
            indent=2,
        )

        sys.stderr.write("\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
