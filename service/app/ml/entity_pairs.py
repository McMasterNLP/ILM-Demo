from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from app.ml.entity_candidates import EntityCandidateService
from app.ml.islamic_entity_linker import (
    IslamicEntityLinker,
    LinkedEntityMention,
)


@dataclass(frozen=True)
class EntityPair:
    sentence: str

    subject_key: str
    subject_text: str
    subject_type: str
    subject_char_start: int
    subject_char_end: int

    object_key: str
    object_text: str
    object_type: str
    object_char_start: int
    object_char_end: int


class EntityPairGenerator:
    """
    Generate directed relation-extraction candidates from canonical
    Islamic entity mentions.

    Important:
    - Uses canonical linked entities, not raw NER spans.
    - Generates both directions.
    - Does not run relation extraction.
    - Does not assign relation labels.
    - Does not write to the database.
    """

    def __init__(
        self,
        linker: Optional[IslamicEntityLinker] = None,
    ) -> None:
        self.linker = linker or IslamicEntityLinker()

    def link_mentions(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[LinkedEntityMention]:
        """
        Run the canonical Islamic entity linker exactly once.

        This is exposed separately so downstream orchestration can
        retain the linked mentions for KG persistence while using the
        same mentions to construct RE candidate pairs.
        """
        return self.linker.link(
            text,
            story_key=story_key,
        )

    def generate_from_mentions(
        self,
        text: str,
        *,
        mentions: List[LinkedEntityMention],
    ) -> List[EntityPair]:
        """
        Generate directed RE pairs from already-linked mentions.

        No NER or entity linking is performed here.
        """
        pairs: List[EntityPair] = []
        seen = set()

        for subject in mentions:
            for obj in mentions:

                # Do not pair a mention with itself.
                if (
                    subject.char_start == obj.char_start
                    and subject.char_end == obj.char_end
                    and subject.canonical_key == obj.canonical_key
                ):
                    continue

                # We do not need relations between two mentions of
                # the exact same canonical entity.
                if subject.canonical_key == obj.canonical_key:
                    continue

                identity = (
                    subject.canonical_key,
                    subject.char_start,
                    subject.char_end,
                    obj.canonical_key,
                    obj.char_start,
                    obj.char_end,
                )

                if identity in seen:
                    continue

                seen.add(identity)

                pairs.append(
                    EntityPair(
                        sentence=text,

                        subject_key=subject.canonical_key,
                        subject_text=subject.text,
                        subject_type=subject.app_type,
                        subject_char_start=subject.char_start,
                        subject_char_end=subject.char_end,

                        object_key=obj.canonical_key,
                        object_text=obj.text,
                        object_type=obj.app_type,
                        object_char_start=obj.char_start,
                        object_char_end=obj.char_end,
                    )
                )

        return sorted(
            pairs,
            key=lambda pair: (
                pair.subject_char_start,
                pair.object_char_start,
                pair.subject_key,
                pair.object_key,
            ),
        )

    def generate(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[EntityPair]:
        mentions = self.link_mentions(
            text,
            story_key=story_key,
        )

        return self.generate_from_mentions(
            text,
            mentions=mentions,
        )

    def generate_dicts(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[dict]:
        return [
            asdict(pair)
            for pair in self.generate(
                text,
                story_key=story_key,
            )
        ]


__all__ = [
    "EntityPair",
    "EntityPairGenerator",
]
