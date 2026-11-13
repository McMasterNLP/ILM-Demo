import re
from dataclasses import asdict, dataclass
from typing import Iterable

from app.ml.re_candidates import RelationCandidate


@dataclass(frozen=True)
class ResolvedPropheticRelation:
    sentence: str

    subject_key: str
    subject_text: str
    subject_type: str

    predicate: str

    object_key: str
    object_text: str
    object_type: str

    resolution_rule: str
    source: str


class PropheticRelationResolver:
    """
    Resolve Islamic-story relations that are outside the WOJOOD
    40-relation source ontology.

    This layer is intentionally separate from the trained RE system.

    Current supported educational relation:
        sent_to

    Verified pattern:
        أرسل الله <PERSON> إلى <GROUP/PLACE>
    """

    SENT_PREFIX = re.compile(
        r"(?:^|\s)"
        r"أرسل\s+الله\s*$"
    )

    SENT_BETWEEN = re.compile(
        r"^\s*إلى\s*$"
    )

    def _unique_mentions(
        self,
        candidates: Iterable[RelationCandidate],
    ) -> list[tuple[str, str, str, int, int]]:
        """
        Recover the unique linked mentions represented by the directed
        Stage-A pair candidates.
        """
        mentions = {}

        for candidate in candidates:
            subject = (
                candidate.subject_key,
                candidate.subject_text,
                candidate.subject_type,
                candidate.subject_char_start,
                candidate.subject_char_end,
            )

            object_ = (
                candidate.object_key,
                candidate.object_text,
                candidate.object_type,
                candidate.object_char_start,
                candidate.object_char_end,
            )

            for mention in (subject, object_):
                key = (
                    mention[0],
                    mention[3],
                    mention[4],
                )
                mentions[key] = mention

        return sorted(
            mentions.values(),
            key=lambda item: (
                item[3],
                item[4],
                item[0],
            ),
        )

    def resolve(
        self,
        *,
        sentence: str,
        candidates: Iterable[RelationCandidate],
    ) -> list[ResolvedPropheticRelation]:

        mentions = self._unique_mentions(candidates)

        results = {}

        for subject in mentions:
            (
                subject_key,
                subject_text,
                subject_type,
                subject_start,
                subject_end,
            ) = subject

            if subject_type != "person":
                continue

            prefix = sentence[:subject_start]

            if self.SENT_PREFIX.search(prefix) is None:
                continue

            for object_ in mentions:
                (
                    object_key,
                    object_text,
                    object_type,
                    object_start,
                    object_end,
                ) = object_

                if subject_key == object_key:
                    continue

                if object_start <= subject_end:
                    continue

                if object_type not in {
                    "group",
                    "place",
                }:
                    continue

                between = sentence[
                    subject_end:object_start
                ]

                if self.SENT_BETWEEN.fullmatch(between) is None:
                    continue

                relation = ResolvedPropheticRelation(
                    sentence=sentence,

                    subject_key=subject_key,
                    subject_text=subject_text,
                    subject_type=subject_type,

                    predicate="sent_to",

                    object_key=object_key,
                    object_text=object_text,
                    object_type=object_type,

                    resolution_rule=(
                        "allah_sent_person_to_group_or_place"
                    ),
                    source="educational_pattern",
                )

                key = (
                    relation.subject_key,
                    relation.predicate,
                    relation.object_key,
                )

                results[key] = relation

        return sorted(
            results.values(),
            key=lambda item: (
                item.subject_key,
                item.predicate,
                item.object_key,
            ),
        )

    def resolve_dicts(
        self,
        *,
        sentence: str,
        candidates: Iterable[RelationCandidate],
    ) -> list[dict]:
        return [
            asdict(item)
            for item in self.resolve(
                sentence=sentence,
                candidates=candidates,
            )
        ]


__all__ = [
    "ResolvedPropheticRelation",
    "PropheticRelationResolver",
]
