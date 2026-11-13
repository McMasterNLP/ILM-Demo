from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import unicodedata

from app.ml.ner_inference import (
    ENTITY_ORDER,
    MODEL_ORDER,
    NERInferenceService,
    _apply_safe_xlmr_additions,
    _build_primary_consensus,
    _get_supporters,
)


Span = Tuple[str, int, int]


# Broad application-level type hints only.
#
# These are NOT final entity-linking decisions. The later Islamic entity
# linker may correct them using canonical knowledge and context.
APP_TYPE_HINTS = {
    "PERS": "person",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "NORP": "group",
}


@dataclass(frozen=True)
class EntityCandidate:
    text: str
    normalized_text: str
    ner_type: str
    app_type_hint: Optional[str]

    token_start: int
    token_end: int
    char_start: int
    char_end: int

    supporters: Tuple[str, ...]
    support_count: int

    # True only when the span survives the exact official final
    # zero-KONOOZ NER ensemble.
    official_ner: bool


def normalize_surface(text: str) -> str:
    """
    Normalize a mention surface for downstream entity linking.

    Only surrounding whitespace and punctuation/symbol characters are
    removed. Internal Arabic text is preserved.

    Model token and character offsets remain untouched elsewhere.
    """

    value = text.strip()

    while value and (
        unicodedata.category(value[0]).startswith("P")
        or unicodedata.category(value[0]).startswith("S")
    ):
        value = value[1:]

    while value and (
        unicodedata.category(value[-1]).startswith("P")
        or unicodedata.category(value[-1]).startswith("S")
    ):
        value = value[:-1]

    return value.strip()


class EntityCandidateService:
    """
    Bridge between the final research NER system and the Islamic-story
    entity-linking layer.

    Important separation:

    1. official_ner=True:
       The exact final research ensemble accepted this span.

    2. official_ner=False:
       At least one individual NER expert proposed the span, but the
       official ensemble did not retain it.

    Raw expert evidence is intentionally preserved because Islamic
    narrative entity linking may recover useful mentions that the
    general-domain ensemble merges, types differently, or suppresses.

    This service performs NO database writes and NO canonical entity
    linking.
    """

    def __init__(
        self,
        ner_service: Optional[NERInferenceService] = None,
    ) -> None:
        self.ner = ner_service or NERInferenceService()

    def _run_experts(
        self,
        tokens: Sequence[str],
    ) -> Dict[str, Set[Span]]:
        """
        Run the exact five inference components already loaded by the
        official NER service.

        The first four are the primary ensemble experts.
        XLM-R remains complementary and is not promoted to an equal fifth
        primary voter.
        """

        return {
            "AraBERT": self.ner._predict_primary_token_expert(
                "AraBERT",
                tokens,
            ),
            "CAMeLBERT": self.ner._predict_primary_token_expert(
                "CAMeLBERT",
                tokens,
            ),
            "MARBERTv2": self.ner._predict_primary_token_expert(
                "MARBERTv2",
                tokens,
            ),
            "LIPN": self.ner._predict_span_expert(tokens),
            "XLMR": self.ner._predict_xlmr(tokens),
        }

    def predict(
        self,
        text: str,
    ) -> List[EntityCandidate]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return []

        tokens, offsets = self.ner._tokenize_with_offsets(text)

        if not tokens:
            return []

        model_spans = self._run_experts(tokens)

        # Reproduce the exact official research ensemble using the
        # implementation already defined in ner_inference.py.
        baseline = _build_primary_consensus(model_spans)

        official_spans = _apply_safe_xlmr_additions(
            baseline,
            model_spans,
        )

        # Candidate layer is deliberately broader than official NER:
        # retain every exact span proposed by at least one expert.
        #
        # We do not resolve these here. Resolution belongs to the later
        # Islamic entity-linking/canonicalization layer.
        raw_spans: Set[Span] = set()

        for spans in model_spans.values():
            raw_spans.update(spans)

        all_model_order = MODEL_ORDER + ["XLMR"]

        candidates: List[EntityCandidate] = []

        for entity_type, start, end in sorted(
            raw_spans,
            key=lambda span: (
                span[1],
                span[2],
                ENTITY_ORDER.index(span[0]),
            ),
        ):
            if (
                start < 0
                or end <= start
                or end > len(tokens)
            ):
                raise ValueError(
                    f"Invalid candidate span: "
                    f"{(entity_type, start, end)}"
                )

            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            surface = text[char_start:char_end]

            supporters = _get_supporters(
                (
                    entity_type,
                    start,
                    end,
                ),
                model_spans,
                all_model_order,
            )

            candidates.append(
                EntityCandidate(
                    text=surface,
                    normalized_text=normalize_surface(
                        surface
                    ),
                    ner_type=entity_type,
                    app_type_hint=APP_TYPE_HINTS.get(
                        entity_type
                    ),
                    token_start=start,
                    token_end=end,
                    char_start=char_start,
                    char_end=char_end,
                    supporters=supporters,
                    support_count=len(supporters),
                    official_ner=(
                        entity_type,
                        start,
                        end,
                    ) in official_spans,
                )
            )

        return candidates

    def predict_dicts(
        self,
        text: str,
    ) -> List[dict]:
        return [
            asdict(candidate)
            for candidate in self.predict(text)
        ]

    def predict_official(
        self,
        text: str,
    ) -> List[EntityCandidate]:
        """
        Convenience view containing only the unchanged official NER
        predictions.
        """

        return [
            candidate
            for candidate in self.predict(text)
            if candidate.official_ner
        ]


__all__ = [
    "APP_TYPE_HINTS",
    "EntityCandidate",
    "EntityCandidateService",
    "normalize_surface",
]
