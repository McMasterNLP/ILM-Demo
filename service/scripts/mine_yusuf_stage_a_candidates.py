#!/usr/bin/env python3

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(
    "/u40/seroura/muslims_ml_project/"
    "dialect-aware-QA-chat-system-main-backup"
)

SERVICE_ROOT = PROJECT_ROOT / "service"

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.ml.re_inference import RelationExistenceEnsemble


QURAN_PATH = (
    SERVICE_ROOT
    / "data"
    / "quran"
    / "derived"
    / "surah_yusuf_simple_clean.jsonl"
)

EVAL_PATH = (
    SERVICE_ROOT
    / "data"
    / "re_adaptation"
    / "yusuf_stage_a_eval_v0.jsonl"
)

TRAIN_PATH = (
    SERVICE_ROOT
    / "data"
    / "re_adaptation"
    / "yusuf_stage_a_train_pool_v0.jsonl"
)

OUTPUT_PATH = (
    SERVICE_ROOT
    / "data"
    / "re_adaptation"
    / "yusuf_stage_a_candidates_v0.tsv"
)


# Conservative mention inventory.
#
# These are explicit surface strings that may occur in Surah Yusuf.
# The miner does NOT assign relation labels.
#
# canonical_id is only for grouping/review.
# entity_type is a coarse application-side type.
MENTIONS = [
    # People
    {
        "surface": "يوسف",
        "canonical_id": "yusuf",
        "entity_type": "person",
    },
    {
        "surface": "يعقوب",
        "canonical_id": "yaqub",
        "entity_type": "person",
    },
    {
        "surface": "إبراهيم",
        "canonical_id": "ibrahim",
        "entity_type": "person",
    },
    {
        "surface": "إسحاق",
        "canonical_id": "ishaq",
        "entity_type": "person",
    },
    {
        "surface": "العزيز",
        "canonical_id": "al_aziz",
        "entity_type": "person",
    },
    {
        "surface": "الملك",
        "canonical_id": "king",
        "entity_type": "person",
    },
    {
        "surface": "امرأت العزيز",
        "canonical_id": "al_aziz_wife",
        "entity_type": "person",
    },
    {
        "surface": "امرأته",
        "canonical_id": "al_aziz_wife",
        "entity_type": "person",
    },

    # Yaqub surface references.
    #
    # We keep these because the original RE model requires literal strings
    # from the sentence. Their canonical identity is recorded separately.
    {
        "surface": "أبيه",
        "canonical_id": "yaqub",
        "entity_type": "person",
    },
    {
        "surface": "أبوه",
        "canonical_id": "yaqub",
        "entity_type": "person",
    },
    {
        "surface": "أبينا",
        "canonical_id": "yaqub",
        "entity_type": "person",
    },
    {
        "surface": "أبانا",
        "canonical_id": "yaqub",
        "entity_type": "person",
    },
    {
        "surface": "أباهم",
        "canonical_id": "yaqub",
        "entity_type": "person",
    },

    # Yusuf's brothers / groups
    {
        "surface": "إخوته",
        "canonical_id": "yusuf_brothers",
        "entity_type": "group",
    },
    {
        "surface": "إخوة",
        "canonical_id": "yusuf_brothers",
        "entity_type": "group",
    },
    {
        "surface": "أخوه",
        "canonical_id": "yusuf_brother",
        "entity_type": "person",
    },
    {
        "surface": "إخوته",
        "canonical_id": "yusuf_brothers",
        "entity_type": "group",
    },
    {
        "surface": "النسوة",
        "canonical_id": "women",
        "entity_type": "group",
    },
    {
        "surface": "نسوة",
        "canonical_id": "women",
        "entity_type": "group",
    },
    {
        "surface": "فتيان",
        "canonical_id": "two_young_men",
        "entity_type": "group",
    },
    {
        "surface": "الملأ",
        "canonical_id": "court_group",
        "entity_type": "group",
    },
    {
        "surface": "الناس",
        "canonical_id": "people",
        "entity_type": "group",
    },
    {
        "surface": "السيارة",
        "canonical_id": "caravan",
        "entity_type": "group",
    },
    {
        "surface": "الرسول",
        "canonical_id": "messenger",
        "entity_type": "person",
    },

    # Places
    {
        "surface": "مصر",
        "canonical_id": "egypt",
        "entity_type": "place",
    },
    {
        "surface": "المدينة",
        "canonical_id": "city",
        "entity_type": "place",
    },
    {
        "surface": "السجن",
        "canonical_id": "prison",
        "entity_type": "place",
    },
    {
        "surface": "الجب",
        "canonical_id": "well",
        "entity_type": "place",
    },
]


def load_jsonl(path):
    if not path.exists():
        return []

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def annotation_key(row):
    return (
        int(row["surah"]),
        int(row["ayah"]),
        row["subject"],
        row["object"],
    )


def reverse_annotation_key(row):
    return (
        int(row["surah"]),
        int(row["ayah"]),
        row["object"],
        row["subject"],
    )


def find_mentions(text):
    found = []

    seen = set()

    for mention in MENTIONS:
        surface = mention["surface"]

        if surface not in text:
            continue

        key = (
            surface,
            mention["canonical_id"],
        )

        if key in seen:
            continue

        seen.add(key)

        found.append(
            {
                **mention,
                "start": text.find(surface),
            }
        )

    # Textual order helps make candidate review easier.
    found.sort(
        key=lambda item: (
            item["start"],
            -len(item["surface"]),
            item["surface"],
        )
    )

    return found


def main():
    ayat = load_jsonl(QURAN_PATH)

    existing_rows = (
        load_jsonl(EVAL_PATH)
        + load_jsonl(TRAIN_PATH)
    )

    used_pairs = set()

    for row in existing_rows:
        used_pairs.add(annotation_key(row))
        used_pairs.add(reverse_annotation_key(row))

    candidates = []

    for ayah_row in ayat:
        surah = int(ayah_row["surah"])
        ayah = int(ayah_row["ayah"])
        sentence = ayah_row["text_ar"]

        mentions = find_mentions(sentence)

        if len(mentions) < 2:
            continue

        for subject, object_ in combinations(mentions, 2):

            # Skip two surface forms resolving to the same entity.
            if (
                subject["canonical_id"]
                == object_["canonical_id"]
            ):
                continue

            pair_key = (
                surah,
                ayah,
                subject["surface"],
                object_["surface"],
            )

            reverse_key = (
                surah,
                ayah,
                object_["surface"],
                subject["surface"],
            )

            # Never mine something already manually annotated.
            if pair_key in used_pairs or reverse_key in used_pairs:
                continue

            candidates.append(
                {
                    "surah": surah,
                    "ayah": ayah,
                    "sentence": sentence,
                    "subject": subject["surface"],
                    "subject_canonical": subject["canonical_id"],
                    "subject_type": subject["entity_type"],
                    "object": object_["surface"],
                    "object_canonical": object_["canonical_id"],
                    "object_type": object_["entity_type"],
                }
            )

    print(
        f"Unscored candidate pairs: {len(candidates)}"
    )

    if not candidates:
        print("No candidate pairs found.")
        return

    print("Loading Stage-A ensemble...")
    model = RelationExistenceEnsemble()

    scored = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        probability = model.predict_probability(
            sentence=candidate["sentence"],
            subject=candidate["subject"],
            object_=candidate["object"],
        )

        candidate["existence_probability"] = probability
        candidate["current_prediction"] = (
            "relation"
            if probability >= 0.5
            else "no_relation"
        )

        scored.append(candidate)

        print(
            f"[{index:03d}/{len(candidates):03d}] "
            f"12:{candidate['ayah']:03d} "
            f"{candidate['subject']} -> "
            f"{candidate['object']} "
            f"P={probability:.6f}"
        )

    # Most suspicious possible hard negatives first:
    # current model is most confident a relation exists.
    scored.sort(
        key=lambda row: (
            -row["existence_probability"],
            row["ayah"],
            row["subject"],
            row["object"],
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "rank",
        "surah",
        "ayah",
        "subject",
        "subject_canonical",
        "subject_type",
        "object",
        "object_canonical",
        "object_type",
        "existence_probability",
        "current_prediction",
        "gold_relation",
        "review_status",
        "sentence",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        for rank, row in enumerate(
            scored,
            start=1,
        ):
            writer.writerow(
                {
                    "rank": rank,
                    **row,
                    "existence_probability": (
                        f"{row['existence_probability']:.6f}"
                    ),
                    "gold_relation": "",
                    "review_status": "",
                }
            )

    print()
    print("Saved:")
    print(OUTPUT_PATH)
    print()
    print("Top 20 candidates:")
    print("-" * 100)

    for rank, row in enumerate(
        scored[:20],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"12:{row['ayah']:03d} | "
            f"P={row['existence_probability']:.6f} | "
            f"{row['subject']} -> {row['object']}"
        )


if __name__ == "__main__":
    main()
