from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import re

from app.ml.entity_candidates import (
    EntityCandidate,
    EntityCandidateService,
    normalize_surface,
)


# ============================================================================
# Canonical Islamic-story entity inventory
#
# IMPORTANT:
# - Canonical keys follow the application's existing Entity.key convention.
# - Ambiguous aliases are STORY-SCOPED.
# - This is entity linking / augmentation, NOT a modification of research NER.
# - No relation knowledge is encoded here.
# ============================================================================

ENTITY_INVENTORY: Dict[
    str,
    List[Tuple[str, str, str]],
] = {
    "yusuf": [
        ("إبراهيم", "ibrahim", "person"),
        ("إسحاق", "ishaq", "person"),
        ("يعقوب", "yaqub", "person"),
        ("يوسف", "yusuf", "person"),

        ("إخوة يوسف", "brothers", "group"),
        ("إخوته", "brothers", "group"),
        ("إخوته", "brothers", "group"),

        ("العزيز", "alaziz", "person"),
        ("زليخا", "zulaikha", "person"),

        ("البئر", "well", "place"),
        ("بئر", "well", "place"),
        ("مصر", "egypt", "place"),
        ("السجن", "prison", "place"),
        ("سجن", "prison", "place"),
    ],

    "nuh": [
        ("نوحا", "nuh", "person"),
        ("نوح", "nuh", "person"),

        ("ابنه", "nuh_son", "person"),

        ("قوم نوح", "nuh_people", "group"),
        ("قومه", "nuh_people", "group"),

        ("الملأ", "nuh_elite", "group"),
    ],

    "musa": [
        ("موسى", "musa", "person"),
        ("هارون", "harun", "person"),

        ("فرعون", "pharaoh", "person"),

        ("آل فرعون", "pharaoh_people", "group"),

        ("بني إسرائيل", "children_of_israel", "group"),
        ("بنو إسرائيل", "children_of_israel", "group"),

        ("السحرة", "magicians", "group"),
        ("الملأ", "pharaoh_elite", "group"),

        ("أمه", "musa_mother", "person"),
        ("أم موسى", "musa_mother", "person"),
        ("أخته", "musa_sister", "person"),

        ("امرأة فرعون", "pharaoh_wife", "person"),

        ("مدين", "madyan", "place"),
        ("مصر", "egypt", "place"),
    ],

    "yunus": [
        ("يونس", "yunus", "person"),
        ("ذا النون", "yunus", "person"),
        ("ذي النون", "yunus", "person"),

        ("قوم يونس", "yunus_people", "group"),
        ("قومه", "yunus_people", "group"),
    ],

    "ibrahim": [
        ("إبراهيم", "ibrahim", "person"),

        ("آزر", "azar", "person"),
        ("أبيه", "azar", "person"),

        ("إسحاق", "ishaq", "person"),
        ("يعقوب", "yaqub", "person"),

        ("لوطا", "lut", "person"),
        ("لوط", "lut", "person"),

        ("قومه", "ibrahim_people", "group"),
        ("قومك", "ibrahim_people", "group"),
    ],

    "salih": [
        ("صالحا", "salih", "person"),
        ("صالح", "salih", "person"),

        ("ثمود", "thamud", "group"),
        ("قومه", "thamud", "group"),
    ],
}


# Arabic proclitics already validated in the Prophet candidate miner.
ALLOWED_PREFIXES = {
    "",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
    "وب",
    "ول",
    "وك",
    "فب",
    "فل",
    "فك",
}


ARABIC_CHAR_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
)


@dataclass(frozen=True)
class LinkedEntityMention:
    canonical_key: str
    app_type: str

    text: str
    normalized_text: str
    matched_alias: str

    char_start: int
    char_end: int

    # Exact NER candidate evidence for this precise character span.
    exact_ner_types: Tuple[str, ...]
    exact_supporters: Tuple[str, ...]

    # NER evidence whose span overlaps this canonical mention.
    # This lets us preserve evidence for cases such as:
    #
    #   "أبا إسحاق" predicted by NER
    #        but
    #   "إسحاق" recovered by canonical linking.
    overlap_ner_types: Tuple[str, ...]
    overlap_supporters: Tuple[str, ...]

    # True only when the exact canonical mention itself was accepted
    # by the unchanged official NER ensemble.
    official_ner: bool

    # How this mention entered the downstream entity layer.
    sources: Tuple[str, ...]


def _is_arabic_char(char: str) -> bool:
    return bool(
        char
        and ARABIC_CHAR_PATTERN.fullmatch(char)
    )


def _valid_occurrence(
    text: str,
    start: int,
    end: int,
) -> bool:
    """
    Apply the same Arabic-token boundary policy used in the validated
    Prophet candidate miner.

    Exact tokens and recognized attached proclitics are accepted.

    Example:
        وموسى   -> accepts موسى
        بإبراهيم -> accepts إبراهيم

    But substring matches inside unrelated Arabic words are rejected.
    """

    if (
        end < len(text)
        and _is_arabic_char(text[end])
    ):
        return False

    token_start = start

    while (
        token_start > 0
        and _is_arabic_char(
            text[token_start - 1]
        )
    ):
        token_start -= 1

    prefix = text[token_start:start]

    return prefix in ALLOWED_PREFIXES


def _find_alias_occurrences(
    text: str,
    alias: str,
) -> List[Tuple[int, int]]:
    occurrences: List[Tuple[int, int]] = []

    search_from = 0

    while True:
        start = text.find(
            alias,
            search_from,
        )

        if start < 0:
            break

        end = start + len(alias)

        if _valid_occurrence(
            text,
            start,
            end,
        ):
            occurrences.append(
                (start, end)
            )

        search_from = start + 1

    return occurrences


def _char_overlaps(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return (
        first_start < second_end
        and second_start < first_end
    )


def _ordered_unique(
    values: Sequence[str],
) -> Tuple[str, ...]:
    result: List[str] = []

    for value in values:
        if value not in result:
            result.append(value)

    return tuple(result)


class IslamicEntityLinker:
    """
    Story-scoped canonical entity linker for Islamic narratives.

    This layer intentionally sits AFTER the research NER system.

    It does not alter NER predictions.

    Instead it:
      1. runs the existing EntityCandidateService;
      2. finds canonical Islamic-story entities directly in the text;
      3. attaches exact and overlapping NER evidence;
      4. recovers canonical entities that NER missed or merged.

    It performs NO database writes.
    """

    def __init__(
        self,
        candidate_service: Optional[
            EntityCandidateService
        ] = None,
    ) -> None:
        self.candidates = (
            candidate_service
            or EntityCandidateService()
        )

    def available_stories(self) -> Tuple[str, ...]:
        return tuple(
            ENTITY_INVENTORY.keys()
        )

    def _inventory_for_story(
        self,
        story_key: str,
    ) -> List[Tuple[str, str, str]]:
        if story_key not in ENTITY_INVENTORY:
            raise ValueError(
                "Unknown story_key "
                f"{story_key!r}. "
                "Available stories: "
                f"{', '.join(self.available_stories())}"
            )

        return ENTITY_INVENTORY[
            story_key
        ]

    def link(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[LinkedEntityMention]:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not isinstance(story_key, str):
            raise TypeError(
                "story_key must be a string"
            )

        if not text.strip():
            return []

        inventory = self._inventory_for_story(
            story_key
        )

        ner_candidates = (
            self.candidates.predict(text)
        )

        linked: List[
            LinkedEntityMention
        ] = []

        seen = set()

        # Longer aliases first only makes output easier to inspect.
        # Nested canonical mentions are still allowed.
        ordered_inventory = sorted(
            inventory,
            key=lambda item: (
                -len(item[0]),
                item[1],
                item[0],
            ),
        )

        for (
            alias,
            canonical_key,
            app_type,
        ) in ordered_inventory:

            for (
                char_start,
                char_end,
            ) in _find_alias_occurrences(
                text,
                alias,
            ):
                identity = (
                    canonical_key,
                    char_start,
                    char_end,
                )

                if identity in seen:
                    continue

                seen.add(identity)

                exact_candidates = [
                    candidate
                    for candidate
                    in ner_candidates
                    if (
                        candidate.char_start
                        == char_start
                        and candidate.char_end
                        == char_end
                    )
                ]

                overlap_candidates = [
                    candidate
                    for candidate
                    in ner_candidates
                    if _char_overlaps(
                        char_start,
                        char_end,
                        candidate.char_start,
                        candidate.char_end,
                    )
                ]

                exact_ner_types = (
                    _ordered_unique(
                        [
                            candidate.ner_type
                            for candidate
                            in exact_candidates
                        ]
                    )
                )

                exact_supporters = (
                    _ordered_unique(
                        [
                            supporter
                            for candidate
                            in exact_candidates
                            for supporter
                            in candidate.supporters
                        ]
                    )
                )

                overlap_ner_types = (
                    _ordered_unique(
                        [
                            candidate.ner_type
                            for candidate
                            in overlap_candidates
                        ]
                    )
                )

                overlap_supporters = (
                    _ordered_unique(
                        [
                            supporter
                            for candidate
                            in overlap_candidates
                            for supporter
                            in candidate.supporters
                        ]
                    )
                )

                official_ner = any(
                    candidate.official_ner
                    for candidate
                    in exact_candidates
                )

                sources: List[str] = [
                    "canonical_lexicon"
                ]

                if exact_candidates:
                    sources.append(
                        "exact_ner_evidence"
                    )
                elif overlap_candidates:
                    sources.append(
                        "overlap_ner_evidence"
                    )

                if official_ner:
                    sources.append(
                        "official_ner"
                    )

                surface = text[
                    char_start:char_end
                ]

                linked.append(
                    LinkedEntityMention(
                        canonical_key=canonical_key,
                        app_type=app_type,
                        text=surface,
                        normalized_text=(
                            normalize_surface(
                                surface
                            )
                        ),
                        matched_alias=alias,
                        char_start=char_start,
                        char_end=char_end,
                        exact_ner_types=(
                            exact_ner_types
                        ),
                        exact_supporters=(
                            exact_supporters
                        ),
                        overlap_ner_types=(
                            overlap_ner_types
                        ),
                        overlap_supporters=(
                            overlap_supporters
                        ),
                        official_ner=(
                            official_ner
                        ),
                        sources=tuple(
                            sources
                        ),
                    )
                )

        return sorted(
            linked,
            key=lambda mention: (
                mention.char_start,
                mention.char_end,
                mention.canonical_key,
            ),
        )

    def link_dicts(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[dict]:
        return [
            asdict(mention)
            for mention
            in self.link(
                text,
                story_key=story_key,
            )
        ]


__all__ = [
    "ENTITY_INVENTORY",
    "IslamicEntityLinker",
    "LinkedEntityMention",
]
