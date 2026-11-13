from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from app.ml.islamic_entity_linker import LinkedEntityMention


@dataclass(frozen=True)
class ResolvedMajorEventRelation:
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


class MajorEventRelationResolver:
    """
    Resolve a small controlled set of major narrative events.

    This layer is separate from the trained WOJOOD RE model and only
    creates an educational relation when an explicit supported Arabic
    construction and the required linked entities are both present.
    """

    #
    # Direct construction where Yusuf is the grammatical object:
    #
    #     ... threw him into a/the well
    #
    # Deliberately do NOT accept bare "threw" / "they threw", because the
    # object may be something else (for example, a caravan throwing a rope
    # into the well).
    #
    THROWN_INTO = re.compile(
        r"(?:"
        r"\u0623\u0644\u0642\u0648\u0647"
        r"|\u0641\u0623\u0644\u0642\u0648\u0647"
        r"|\u064a\u0644\u0642\u0648\u0647"
        r")"
        r".{0,80}?"
        r"\u0641\u064a\s*"
        r"(?:\u0627\u0644)?\u0628\u0626\u0631",
        re.DOTALL,
    )

    #
    # Anaphoric construction used in the curated Yusuf story:
    #
    #     ... a deep well ... they threw him in it
    #
    # The linked well must occur before the pronoun-bearing throw verb.
    #
    THROWN_INTO_REFERENCED_WELL = re.compile(
        r"(?:\u0627\u0644)?\u0628\u0626\u0631"
        r".{0,100}?"
        r"(?:"
        r"\u0623\u0644\u0642\u0648\u0647"
        r"|\u0641\u0623\u0644\u0642\u0648\u0647"
        r")"
        r"\s+\u0641\u064a\u0647",
        re.DOTALL,
    )

    TAKEN_TO = re.compile(
        r"(?:"
        r"\u0623\u062e\u0630\u0648\u0647"
        r"|\u0623\u062e\u0630\u0648\u0627\s+\u064a\u0648\u0633\u0641"
        r"|\u062d\u0645\u0644\u0648\u0647"
        r"|\u0646\u0642\u0644\u0648\u0647"
        r"|\u0630\u0647\u0628\s+\u0628\u0647"
        r"|\u062c\u064a\u0621\s+\u0628\u0647"
        r"|\u0627\u0646\u062a\u0642\u0644\s+\u064a\u0648\u0633\u0641"
        r")"
        r".{0,140}?"
        r"(?:"
        r"\u0625\u0644\u0649\s+\u0645\u0635\u0631"
        r"|\u0641\u064a\s+\u0645\u0635\u0631"
        r"|\u0645\u0635\u0631"
        r")",
        re.DOTALL,
    )

    IMPRISONED_IN = re.compile(
        r"(?:"
        r"\u062f\u062e\u0644"
        r"|\u064a\u062f\u062e\u0644"
        r"|\u0623\u062f\u062e\u0644"
        r"|\u0623\u0648\u062f\u0639"
        r"|\u0633\u062c\u0646"
        r"|\u0633\u064f\u062c\u0646"
        r")"
        r".{0,80}?"
        r"(?:\u0627\u0644)?\u0633\u062c\u0646",
        re.DOTALL,
    )

    APPOINTED_OVER = re.compile(
        r"(?:"
        r"\u0627\u0633\u062a\u0639\u0645\u0644\u0647\s+\u0627\u0644\u0645\u0644\u0643"
        r"|\u0648\u0644\u0627\u0647\s+\u0627\u0644\u0645\u0644\u0643"
        r"|\u0648\u0644\u0651\u0627\u0647\s+\u0627\u0644\u0645\u0644\u0643"
        r"|\u064a\u0634\u0631\u0641"
        r"|\u064a\u062f\u064a\u0631"
        r")"
        r".{0,160}?"
        r"(?:"
        r"\u0627\u0642\u062a\u0635\u0627\u062f\s+\u0645\u0635\u0631"
        r"|\u0645\u0648\u0627\u0631\u062f\s+(?:\u0627\u0644\u062f\u0648\u0644\u0629|\u0645\u0635\u0631)"
        r"|\u0634\u0624\u0648\u0646\s+\u0645\u0635\u0631"
        r"|\u0645\u0635\u0631"
        r")",
        re.DOTALL,
    )

    REUNITED_WITH = re.compile(
        r"(?:"
        r"\u0627\u062c\u062a\u0645\u0639"
        r"|\u0627\u0644\u062a\u0642\u0649"
        r"|\u064a\u0644\u062a\u0626\u0645\s+\u0634\u0645\u0644"
        r"|\u0648\u0635\u0644\s+\u064a\u0639\u0642\u0648\u0628"
        r"|\u064a\u0635\u0644\s+\u064a\u0639\u0642\u0648\u0628"
        r")",
        re.DOTALL,
    )

    @staticmethod
    def _unique_mentions(
        mentions: Iterable["LinkedEntityMention"],
    ) -> list["LinkedEntityMention"]:
        unique = {}

        for mention in mentions:
            key = (
                mention.canonical_key,
                mention.char_start,
                mention.char_end,
            )
            unique[key] = mention

        return sorted(
            unique.values(),
            key=lambda item: (
                item.char_start,
                item.char_end,
                item.canonical_key,
            ),
        )

    @staticmethod
    def _find_by_key(
        mentions: Iterable["LinkedEntityMention"],
        key: str,
    ) -> list["LinkedEntityMention"]:
        return [
            mention
            for mention in mentions
            if mention.canonical_key == key
        ]

    @staticmethod
    def _make(
        *,
        sentence: str,
        subject: "LinkedEntityMention",
        predicate: str,
        object_: "LinkedEntityMention",
        rule: str,
    ) -> ResolvedMajorEventRelation:
        return ResolvedMajorEventRelation(
            sentence=sentence,
            subject_key=subject.canonical_key,
            subject_text=subject.text,
            subject_type=subject.app_type,
            predicate=predicate,
            object_key=object_.canonical_key,
            object_text=object_.text,
            object_type=object_.app_type,
            resolution_rule=rule,
            source="educational_major_event_pattern",
        )

    def resolve(
        self,
        *,
        sentence: str,
        mentions: Iterable["LinkedEntityMention"],
        story_key: str | None = None,
    ) -> list[ResolvedMajorEventRelation]:

        mentions = self._unique_mentions(mentions)
        results = {}

        yusuf = self._find_by_key(mentions, "yusuf")

        targets = {
            "well": self._find_by_key(mentions, "well"),
            "egypt": self._find_by_key(mentions, "egypt"),
            "prison": self._find_by_key(mentions, "prison"),
            "yaqub": self._find_by_key(mentions, "yaqub"),
        }

        #
        # Yusuf-story contextual recovery.
        #
        # Some curated paragraphs refer to Yusuf only by an attached
        # pronoun, for example:
        #
        #     ?????? ?? ???
        #
        # In that case there is no explicit Yusuf span for the linker to
        # return. We do not perform general coreference here. We only allow
        # the canonical Yusuf story subject when the explicit throw-into-well
        # construction is present and the well itself was linked.
        #
        if (
            not yusuf
            and story_key == "yusuf"
            and targets["well"]
            and (
                self.THROWN_INTO.search(sentence) is not None
                or self.THROWN_INTO_REFERENCED_WELL.search(sentence) is not None
            )
        ):
            well = targets["well"][0]

            relation = ResolvedMajorEventRelation(
                sentence=sentence,
                subject_key="yusuf",
                subject_text="\u064a\u0648\u0633\u0641",
                subject_type="person",
                predicate="thrown_into",
                object_key=well.canonical_key,
                object_text=well.text,
                object_type=well.app_type,
                resolution_rule="yusuf_story_implicit_throw_subject",
                source="educational_major_event_pattern",
            )

            results[
                (
                    relation.subject_key,
                    relation.predicate,
                    relation.object_key,
                )
            ] = relation

        #
        # All other major-event rules still require an explicit linked
        # Yusuf mention.
        #
        if not yusuf:
            return sorted(
                results.values(),
                key=lambda item: (
                    item.subject_key,
                    item.predicate,
                    item.object_key,
                ),
            )

        subject = yusuf[0]

        checks = [
            (
                "thrown_into",
                "well",
                self.THROWN_INTO,
                "explicit_throw_into_well",
            ),
            (
                "taken_to",
                "egypt",
                self.TAKEN_TO,
                "explicit_transport_to_egypt",
            ),
            (
                "imprisoned_in",
                "prison",
                self.IMPRISONED_IN,
                "explicit_yusuf_prison_event",
            ),
            (
                "appointed_over",
                "egypt",
                self.APPOINTED_OVER,
                "explicit_administrative_role_in_egypt",
            ),
            (
                "reunited_with",
                "yaqub",
                self.REUNITED_WITH,
                "explicit_yusuf_yaqub_reunion",
            ),
        ]

        for predicate, target_key, pattern, rule in checks:
            target_mentions = targets[target_key]

            if not target_mentions:
                continue

            if predicate == "thrown_into":
                matched = (
                    self.THROWN_INTO.search(sentence) is not None
                    or self.THROWN_INTO_REFERENCED_WELL.search(sentence) is not None
                )
            else:
                matched = pattern.search(sentence) is not None

            if not matched:
                continue

            relation = self._make(
                sentence=sentence,
                subject=subject,
                predicate=predicate,
                object_=target_mentions[0],
                rule=rule,
            )

            results[
                (
                    relation.subject_key,
                    relation.predicate,
                    relation.object_key,
                )
            ] = relation

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
        mentions: Iterable["LinkedEntityMention"],
    ) -> list[dict]:
        return [
            asdict(item)
            for item in self.resolve(
                sentence=sentence,
                mentions=mentions,
            )
        ]


__all__ = [
    "ResolvedMajorEventRelation",
    "MajorEventRelationResolver",
]
