from dataclasses import asdict, dataclass
from typing import Iterable

from app.ml.re_candidates import RelationCandidate
from app.ml.re_inference import (
    NLIReranker,
    PositiveRelationEnsemble,
)


@dataclass(frozen=True)
class PositiveRelationPrediction:
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

    existence_probability: float

    relation_label: str
    relation_confidence: float

    arabert_probability: float
    nli_probability: float


class PositiveRelationCandidateService:
    """
    Stage-B relation classification for entity pairs that have already
    passed the V6 Stage-A relation-existence gate.

    Pipeline:
        Stage-A PASS
            -> strong 40-way positive RE ensemble
            -> NLI reranking
            -> 0.55 RE + 0.45 NLI blend

    This class intentionally does NOT:
        - rerun Stage A
        - apply educational ontology mapping
        - apply schema penalties yet
        - write anything to the database
    """

    def __init__(self) -> None:
        self.positive = PositiveRelationEnsemble()
        self.nli = NLIReranker(self.positive)

    def predict_one(
        self,
        candidate: RelationCandidate,
    ) -> PositiveRelationPrediction:

        if not candidate.passes_existence_gate:
            raise ValueError(
                "Stage-B received a pair that did not pass "
                "the Stage-A existence gate."
            )

        ranked = self.nli.rerank(
            sentence=candidate.sentence,
            subject=candidate.subject_text,
            object_=candidate.object_text,
        )

        if not ranked:
            raise RuntimeError(
                "NLI reranker returned no relation candidates."
            )

        best = ranked[0]

        return PositiveRelationPrediction(
            sentence=candidate.sentence,

            subject_key=candidate.subject_key,
            subject_text=candidate.subject_text,
            subject_type=candidate.subject_type,
            subject_char_start=candidate.subject_char_start,
            subject_char_end=candidate.subject_char_end,

            object_key=candidate.object_key,
            object_text=candidate.object_text,
            object_type=candidate.object_type,
            object_char_start=candidate.object_char_start,
            object_char_end=candidate.object_char_end,

            existence_probability=(
                candidate.existence_probability
            ),

            relation_label=best.relation_label,
            relation_confidence=(
                best.blended_probability
            ),

            arabert_probability=(
                best.arabert_probability
            ),

            nli_probability=(
                best.nli_probability
            ),
        )

    def predict(
        self,
        candidates: Iterable[RelationCandidate],
    ) -> list[PositiveRelationPrediction]:

        results = []

        for candidate in candidates:
            if not candidate.passes_existence_gate:
                continue

            results.append(
                self.predict_one(candidate)
            )

        return results

    def predict_dicts(
        self,
        candidates: Iterable[RelationCandidate],
    ) -> list[dict]:
        return [
            asdict(item)
            for item in self.predict(candidates)
        ]


__all__ = [
    "PositiveRelationPrediction",
    "PositiveRelationCandidateService",
]
