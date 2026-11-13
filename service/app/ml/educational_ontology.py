from dataclasses import dataclass


@dataclass(frozen=True)
class EducationalRelationDefinition:
    key: str
    category: str
    symmetric: bool
    inverse: str | None
    quiz_enabled: bool
    label_en: str
    label_ar: str
    phrase_en: str | None = None
    phrase_ar: str | None = None


RELATIONS: dict[str, EducationalRelationDefinition] = {
    "child_of": EducationalRelationDefinition(
        "child_of",
        "family",
        False,
        "father_of",
        True,
        "child of",
        "ابن / ابنة",
    ),

    "father_of": EducationalRelationDefinition(
        "father_of",
        "family",
        False,
        "child_of",
        True,
        "father of",
        "والد",
    ),

    "sibling_of": EducationalRelationDefinition(
        "sibling_of",
        "family",
        True,
        "sibling_of",
        True,
        "sibling of",
        "أخ / أخت",
    ),

    "sent_to": EducationalRelationDefinition(
        "sent_to",
        "prophetic",
        False,
        None,
        True,
        "sent to",
        "أُرسل إلى",
    ),

    "thrown_into": EducationalRelationDefinition(
        "thrown_into",
        "major_event",
        False,
        None,
        True,
        "thrown into",
        "أُلقي في",
    ),

    "taken_to": EducationalRelationDefinition(
        "taken_to",
        "major_event",
        False,
        None,
        True,
        "taken to",
        "أُخذ إلى",
    ),

    "imprisoned_in": EducationalRelationDefinition(
        "imprisoned_in",
        "major_event",
        False,
        None,
        True,
        "imprisoned in",
        "سُجن في",
    ),

    "appointed_over": EducationalRelationDefinition(
        "appointed_over",
        "major_event",
        False,
        None,
        True,
        "appointed over",
        "عُيّن على",
    ),

    "reunited_with": EducationalRelationDefinition(
        "reunited_with",
        "major_event",
        True,
        "reunited_with",
        True,
        "reunited with",
        "اجتمع مع",
    ),
}


def get_relation(
    predicate: str,
) -> EducationalRelationDefinition:
    try:
        return RELATIONS[predicate]
    except KeyError as exc:
        raise ValueError(
            f"Unknown educational relation: {predicate}"
        ) from exc


def is_allowed_relation(
    predicate: str,
) -> bool:
    return predicate in RELATIONS


def quiz_relations() -> tuple[str, ...]:
    return tuple(
        key
        for key, relation in RELATIONS.items()
        if relation.quiz_enabled
    )


__all__ = [
    "EducationalRelationDefinition",
    "RELATIONS",
    "get_relation",
    "is_allowed_relation",
    "quiz_relations",
]
