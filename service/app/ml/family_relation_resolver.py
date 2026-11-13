import re
from dataclasses import asdict, dataclass
from typing import Iterable

from app.ml.re_positive_candidates import PositiveRelationPrediction


@dataclass(frozen=True)
class ResolvedFamilyRelation:
    sentence: str

    subject_key: str
    subject_text: str
    subject_type: str

    predicate: str

    object_key: str
    object_text: str
    object_type: str

    source_relation: str
    existence_probability: float
    relation_confidence: float

    resolution_rule: str


@dataclass(frozen=True)
class _SemanticMatch:
    subject_key: str
    subject_text: str
    subject_type: str

    predicate: str

    object_key: str
    object_text: str
    object_type: str

    rule: str


class FamilyRelationResolver:
    """
    Resolve directional family semantics after the source RE system.

    Important:
    - Does NOT change the trained RE prediction.
    - Uses the original sentence plus linked-entity offsets.
    - Only applies to verified Arabic family constructions.
    - Unknown constructions are not guessed.
    """

    FAMILY_RELATIONS = {
        "Family.has_parent",
        "Family.has_sibling",
        "Family.has_spouse",
        "Family.has_relative",
    }

    # child ... parent
    CHILD_TO_PARENT = re.compile(
        r"^\s*(?:هو\s+|هي\s+)?"
        r"(?:ابن|إبن|بن|ابنة|إبنة|بنت)\s*$"
    )

    # father ... child
    FATHER_TO_CHILD = re.compile(
        r"^\s*(?:هو\s+|هي\s+)?"
        r"(?:أب|اب|أبو|ابو|أبا|ابا|والد)\s*$"
    )

    # symmetric sibling relation
    SIBLING = re.compile(
        r"^\s*(?:هو\s+|هي\s+)?"
        r"(?:أخ|اخ|أخا|اخا|أخو|اخو|أخت|اخت|شقيق|شقيقة)\s*$"
    )

    @staticmethod
    def _resolve_named_parent_pattern(
        prediction: PositiveRelationPrediction,
    ) -> _SemanticMatch | None:
        """
        Resolve an already-predicted Family.has_parent relation when the
        explicit named parent construction occurs elsewhere in the same
        sentence/paragraph than the exact mention pair selected by RE.

        Example:
            Yusuf ... min abihi Yaqub

        This does not create a family relation on its own. It is only
        consulted after Stage B has predicted Family.has_parent.
        """

        subject = (
            prediction.subject_key,
            prediction.subject_text,
            prediction.subject_type,
        )
        object_ = (
            prediction.object_key,
            prediction.object_text,
            prediction.object_type,
        )

        parent_marker = (
            r"(?:\u0645\u0646\s+)?"
            r"(?:"
            r"\u0623\u0628\u064a\u0647|"
            r"\u0623\u0628\u064a\u0647\u0627|"
            r"\u0648\u0627\u0644\u062f\u0647|"
            r"\u0648\u0627\u0644\u062f\u0647\u0627"
            r")"
            r"\s+"
        )

        for child, parent in (
            (subject, object_),
            (object_, subject),
        ):
            (
                child_key,
                child_text,
                child_type,
            ) = child

            (
                parent_key,
                parent_text,
                parent_type,
            ) = parent

            pattern = (
                re.escape(child_text)
                + r"[^.!?\u061f\n]{0,120}?"
                + parent_marker
                + re.escape(parent_text)
            )

            if re.search(pattern, prediction.sentence):
                return _SemanticMatch(
                    subject_key=child_key,
                    subject_text=child_text,
                    subject_type=child_type,
                    predicate="child_of",
                    object_key=parent_key,
                    object_text=parent_text,
                    object_type=parent_type,
                    rule="named_parent_marker:possessive_father",
                )

        return None

    @staticmethod
    def _between(
        sentence: str,
        left_end: int,
        right_start: int,
    ) -> str:
        return sentence[left_end:right_start].strip()

    @staticmethod
    def _entity_tuple(
        prediction: PositiveRelationPrediction,
        which: str,
    ) -> tuple[str, str, str, int, int]:
        if which == "subject":
            return (
                prediction.subject_key,
                prediction.subject_text,
                prediction.subject_type,
                prediction.subject_char_start,
                prediction.subject_char_end,
            )

        if which == "object":
            return (
                prediction.object_key,
                prediction.object_text,
                prediction.object_type,
                prediction.object_char_start,
                prediction.object_char_end,
            )

        raise ValueError(which)

    def _ordered_mentions(
        self,
        prediction: PositiveRelationPrediction,
    ):
        subject = self._entity_tuple(prediction, "subject")
        object_ = self._entity_tuple(prediction, "object")

        if subject[3] <= object_[3]:
            return subject, object_

        return object_, subject

    def _resolve_semantics(
        self,
        prediction: PositiveRelationPrediction,
    ) -> _SemanticMatch | None:

        if prediction.relation_label not in self.FAMILY_RELATIONS:
            return None

        if prediction.relation_label == "Family.has_parent":
            named_parent = self._resolve_named_parent_pattern(prediction)
            if named_parent is not None:
                return named_parent

        left, right = self._ordered_mentions(prediction)

        (
            left_key,
            left_text,
            left_type,
            _left_start,
            left_end,
        ) = left

        (
            right_key,
            right_text,
            right_type,
            right_start,
            _right_end,
        ) = right

        between = self._between(
            prediction.sentence,
            left_end,
            right_start,
        )

        #
        # X ابن Y
        # X = child, Y = parent
        #
        if self.CHILD_TO_PARENT.fullmatch(between):
            return _SemanticMatch(
                subject_key=left_key,
                subject_text=left_text,
                subject_type=left_type,
                predicate="child_of",
                object_key=right_key,
                object_text=right_text,
                object_type=right_type,
                rule=f"child_marker:{between}",
            )

        #
        # X أبا Y / X أبو Y / X أب Y
        # X = father, Y = child
        #
        if self.FATHER_TO_CHILD.fullmatch(between):
            return _SemanticMatch(
                subject_key=left_key,
                subject_text=left_text,
                subject_type=left_type,
                predicate="father_of",
                object_key=right_key,
                object_text=right_text,
                object_type=right_type,
                rule=f"father_marker:{between}",
            )

        #
        # X أخا Y / X أخ Y / X أخت Y
        # symmetric relation
        #
        if self.SIBLING.fullmatch(between):
            return _SemanticMatch(
                subject_key=left_key,
                subject_text=left_text,
                subject_type=left_type,
                predicate="sibling_of",
                object_key=right_key,
                object_text=right_text,
                object_type=right_type,
                rule=f"sibling_marker:{between}",
            )

        return None

    def resolve_one(
        self,
        prediction: PositiveRelationPrediction,
    ) -> ResolvedFamilyRelation | None:

        semantic = self._resolve_semantics(prediction)

        if semantic is None:
            return None

        return ResolvedFamilyRelation(
            sentence=prediction.sentence,

            subject_key=semantic.subject_key,
            subject_text=semantic.subject_text,
            subject_type=semantic.subject_type,

            predicate=semantic.predicate,

            object_key=semantic.object_key,
            object_text=semantic.object_text,
            object_type=semantic.object_type,

            source_relation=prediction.relation_label,
            existence_probability=prediction.existence_probability,
            relation_confidence=prediction.relation_confidence,

            resolution_rule=semantic.rule,
        )

    def resolve(
        self,
        predictions: Iterable[PositiveRelationPrediction],
    ) -> list[ResolvedFamilyRelation]:
        """
        Resolve predictions and collapse duplicate semantic edges.

        For Family.has_parent, preserve provenance from the source-RE
        direction that agrees with the source ontology:

            Family.has_parent = child -> parent

        For symmetric family relations, retain the strongest prediction.
        """
        best = {}

        for prediction in predictions:
            resolved = self.resolve_one(prediction)

            if resolved is None:
                continue

            key = (
                resolved.subject_key,
                resolved.predicate,
                resolved.object_key,
            )

            #
            # Directional provenance for Family.has_parent.
            #
            if (
                prediction.relation_label == "Family.has_parent"
                and resolved.predicate == "child_of"
            ):
                #
                # Educational child_of has the same direction as the
                # WOJOOD source relation:
                #
                # child -> parent
                #
                direction_rank = int(
                    prediction.subject_key
                    == resolved.subject_key
                    and prediction.object_key
                    == resolved.object_key
                )

            elif (
                prediction.relation_label == "Family.has_parent"
                and resolved.predicate == "father_of"
            ):
                #
                # Educational father_of reverses the WOJOOD direction.
                #
                # WOJOOD:
                #     child -> parent
                #
                # Educational:
                #     parent -> child
                #
                direction_rank = int(
                    prediction.subject_key
                    == resolved.object_key
                    and prediction.object_key
                    == resolved.subject_key
                )

            else:
                #
                # sibling_of / spouse-style relations are symmetric for
                # this resolver, so confidence can decide between the
                # duplicate directed source predictions.
                #
                direction_rank = 1

            candidate_rank = (
                direction_rank,
                resolved.relation_confidence,
            )

            previous = best.get(key)

            if (
                previous is None
                or candidate_rank > previous[0]
            ):
                best[key] = (
                    candidate_rank,
                    resolved,
                )

        resolved_relations = [
            value[1]
            for value in best.values()
        ]

        return sorted(
            resolved_relations,
            key=lambda item: (
                item.subject_key,
                item.predicate,
                item.object_key,
            ),
        )

    def resolve_dicts(
        self,
        predictions: Iterable[PositiveRelationPrediction],
    ) -> list[dict]:
        return [
            asdict(item)
            for item in self.resolve(predictions)
        ]


__all__ = [
    "ResolvedFamilyRelation",
    "FamilyRelationResolver",
]
