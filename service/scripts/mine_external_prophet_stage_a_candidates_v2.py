#!/usr/bin/env python3

import re
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(
    "/u40/seroura/muslims_ml_project/"
    "dialect-aware-QA-chat-system-main-backup"
)

SERVICE_ROOT = PROJECT_ROOT / "service"

RE_ROOT = Path(
    "/u40/seroura/knowledgegrapheval_re_2026"
)

sys.path.insert(0, str(RE_ROOT / "src"))

from train_marker_fgm_ner_aux_twohead_v1 import (
    MultiTaskRelationModel,
    insert_markers,
)


SOURCE_PATH = (
    SERVICE_ROOT
    / "data/quran/derived/"
    "prophets_external_re_source_v1.jsonl"
)

ADAPTATION_ROOT = (
    SERVICE_ROOT
    / "models/re_adaptation/"
    "yusuf_stage_a_layer11_v1"
)

OUTPUT_TSV = (
    SERVICE_ROOT
    / "data/re_adaptation/"
    "external_prophets_stage_a_candidates_v2.tsv"
)

OUTPUT_JSONL = (
    SERVICE_ROOT
    / "data/re_adaptation/"
    "external_prophets_stage_a_candidates_v2.jsonl"
)

FOLD_DIRS = [
    RE_ROOT
    / "runs"
    / (
        "arabertv02_marker_fgm_neraux015_twohead_oof_v1_"
        f"fold{fold}_seed42"
    )
    / "best_model"
    for fold in range(5)
]

THRESHOLD = 0.5
MAX_LENGTH = 384
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# Conservative entity surface inventory
#
# These are candidate mentions only.
# Nothing here is treated as a gold relation.
# =====================================================================

ENTITY_INVENTORY = {
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


# =====================================================================
# File utilities
# =====================================================================

def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def load_label_metadata(model_dir):
    metadata = json.loads(
        (
            model_dir
            / "training_metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    label_to_id = {
        str(label): int(index)
        for label, index
        in metadata["label_to_id"].items()
    }

    id_to_label = {
        int(index): str(label)
        for index, label
        in metadata["id_to_label"].items()
    }

    return label_to_id, id_to_label


# =====================================================================
# Load one adapted fold
# =====================================================================

def load_fold(fold):
    model_dir = FOLD_DIRS[fold]

    if not model_dir.exists():
        raise FileNotFoundError(
            model_dir
        )

    delta_path = (
        ADAPTATION_ROOT
        / f"fold{fold}"
        / "stage_a_layer11_delta.pt"
    )

    if not delta_path.exists():
        raise FileNotFoundError(
            delta_path
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    label_to_id, id_to_label = (
        load_label_metadata(
            model_dir
        )
    )

    model = MultiTaskRelationModel(
        model_name=str(model_dir),
        num_relation_labels=len(
            label_to_id
        ),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        subject_marker_id=(
            tokenizer.convert_tokens_to_ids(
                "[SUBJ]"
            )
        ),
        object_marker_id=(
            tokenizer.convert_tokens_to_ids(
                "[OBJ]"
            )
        ),
    )

    auxiliary = torch.load(
        model_dir
        / "ner_auxiliary_heads.pt",
        map_location="cpu",
        weights_only=False,
    )

    model.subject_type_classifier.load_state_dict(
        auxiliary[
            "subject_type_classifier"
        ]
    )

    model.object_type_classifier.load_state_dict(
        auxiliary[
            "object_type_classifier"
        ]
    )

    model.existence_classifier.load_state_dict(
        auxiliary[
            "existence_classifier"
        ]
    )

    delta = torch.load(
        delta_path,
        map_location="cpu",
        weights_only=False,
    )

    model.re_model.bert.encoder.layer[
        11
    ].load_state_dict(
        delta[
            "encoder_layer_11"
        ]
    )

    model.existence_classifier.load_state_dict(
        delta[
            "existence_classifier"
        ]
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer


# =====================================================================
# Entity mention mining
# =====================================================================

def find_mentions(
    prophet,
    text,
):
    mentions = []

    inventory = ENTITY_INVENTORY.get(
        prophet,
        [],
    )

    # Arabic letters/marks used only to determine token continuity.
    arabic_char_pattern = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
    )

    # Arabic proclitics that may attach directly to a named entity.
    #
    # Examples:
    #   وموسى  -> موسى
    #   فموسى  -> موسى
    #   بإبراهيم -> إبراهيم
    #   ليعقوب -> يعقوب
    #
    # We deliberately do NOT include the definite article "ال".
    allowed_prefixes = {
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

    def is_arabic_char(char):
        return bool(
            char
            and arabic_char_pattern.fullmatch(char)
        )

    def valid_occurrence(
        start,
        end,
    ):
        # ----------------------------------------------------------
        # Right boundary:
        # the surface must end at an Arabic-word boundary.
        # ----------------------------------------------------------

        if (
            end < len(text)
            and is_arabic_char(text[end])
        ):
            return False

        # ----------------------------------------------------------
        # Left side:
        # find beginning of the complete Arabic token containing
        # this surface.
        # ----------------------------------------------------------

        token_start = start

        while (
            token_start > 0
            and is_arabic_char(
                text[token_start - 1]
            )
        ):
            token_start -= 1

        prefix = text[
            token_start:start
        ]

        # Exact token or recognised attached Arabic proclitic.
        return prefix in allowed_prefixes

    for (
        surface,
        canonical_id,
        entity_type,
    ) in inventory:

        search_start = 0

        while True:
            index = text.find(
                surface,
                search_start,
            )

            if index < 0:
                break

            end = (
                index
                + len(surface)
            )

            if valid_occurrence(
                index,
                end,
            ):
                mentions.append(
                    {
                        "surface": surface,
                        "canonical_id": canonical_id,
                        "entity_type": entity_type,
                        "start": index,
                        "end": end,
                    }
                )

            search_start = (
                index + 1
            )

    # Prefer longer overlapping mentions representing the same
    # canonical entity.
    mentions.sort(
        key=lambda x: (
            x["start"],
            -(x["end"] - x["start"]),
        )
    )

    filtered = []

    for mention in mentions:
        duplicate = False

        for existing in filtered:
            same_entity = (
                mention["canonical_id"]
                == existing["canonical_id"]
            )

            overlaps = (
                mention["start"]
                < existing["end"]
                and mention["end"]
                > existing["start"]
            )

            if (
                same_entity
                and overlaps
            ):
                duplicate = True
                break

        if not duplicate:
            filtered.append(
                mention
            )

    return filtered

def build_candidates(rows):
    candidates = []

    seen = set()

    for row in rows:
        prophet = row["prophet"]
        sentence = row["text_ar"]

        mentions = find_mentions(
            prophet,
            sentence,
        )

        if len(mentions) < 2:
            continue

        for subject in mentions:
            for object_ in mentions:

                if subject is object_:
                    continue

                # Same canonical entity is not an RE pair.
                if (
                    subject["canonical_id"]
                    == object_["canonical_id"]
                ):
                    continue

                key = (
                    prophet,
                    row["surah"],
                    row["ayah"],
                    subject["surface"],
                    subject["canonical_id"],
                    object_["surface"],
                    object_["canonical_id"],
                )

                if key in seen:
                    continue

                seen.add(key)

                marked_sentence, marker_status = (
                    insert_markers(
                        sentence,
                        subject["surface"],
                        object_["surface"],
                    )
                )

                candidates.append(
                    {
                        "prophet": prophet,
                        "surah": row["surah"],
                        "ayah": row["ayah"],

                        "subject": (
                            subject["surface"]
                        ),
                        "subject_canonical": (
                            subject[
                                "canonical_id"
                            ]
                        ),
                        "subject_type": (
                            subject[
                                "entity_type"
                            ]
                        ),

                        "object": (
                            object_["surface"]
                        ),
                        "object_canonical": (
                            object_[
                                "canonical_id"
                            ]
                        ),
                        "object_type": (
                            object_[
                                "entity_type"
                            ]
                        ),

                        "marker_status": (
                            marker_status
                        ),

                        "sentence": sentence,
                    }
                )

    return candidates


# =====================================================================
# Stage-A inference
# =====================================================================

@torch.no_grad()
def score_candidates(
    candidates,
):
    all_fold_probabilities = []

    for fold in range(5):
        print(
            f"Scoring fold {fold}..."
        )

        model, tokenizer = (
            load_fold(fold)
        )

        fold_probabilities = []

        for index, row in enumerate(
            candidates,
            start=1,
        ):
            marked_sentence, marker_status = (
                insert_markers(
                    row["sentence"],
                    row["subject"],
                    row["object"],
                )
            )

            pair_text = (
                f'[SUBJ] {row["subject"]} [/SUBJ] '
                f'[OBJ] {row["object"]} [/OBJ]'
            )

            encoded = tokenizer(
                marked_sentence,
                pair_text,
                truncation="only_first",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value
                in encoded.items()
            }

            outputs = model.re_model.bert(
                **encoded,
                return_dict=True,
            )

            cls_representation = (
                outputs.last_hidden_state[
                    :,
                    0,
                    :,
                ]
            )

            logits = (
                model.existence_classifier(
                    model.existence_dropout(
                        cls_representation
                    )
                )
            )

            probability = (
                torch.softmax(
                    logits.float(),
                    dim=-1,
                )[0, 1]
                .item()
            )

            fold_probabilities.append(
                probability
            )

        all_fold_probabilities.append(
            np.asarray(
                fold_probabilities,
                dtype=np.float64,
            )
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble = np.mean(
        np.stack(
            all_fold_probabilities,
            axis=0,
        ),
        axis=0,
    )

    return ensemble


# =====================================================================
# Save review files
# =====================================================================

def main():
    print(
        "Device:",
        DEVICE,
    )

    print(
        "Source:",
        SOURCE_PATH,
    )

    rows = load_jsonl(
        SOURCE_PATH
    )

    candidates = build_candidates(
        rows
    )

    print(
        "Raw candidate pairs:",
        len(candidates),
    )

    if not candidates:
        raise RuntimeError(
            "No candidate pairs were mined."
        )

    probabilities = score_candidates(
        candidates
    )

    for row, probability in zip(
        candidates,
        probabilities,
    ):
        row[
            "existence_probability"
        ] = float(
            probability
        )

        row[
            "current_prediction"
        ] = (
            "relation"
            if probability >= THRESHOLD
            else "no_relation"
        )

        # Human-review fields intentionally empty.
        row["gold_relation"] = ""
        row["review_status"] = ""

    candidates.sort(
        key=lambda row: (
            -row[
                "existence_probability"
            ],
            row["prophet"],
            row["surah"],
            row["ayah"],
        )
    )

    for rank, row in enumerate(
        candidates,
        start=1,
    ):
        row["rank"] = rank

    OUTPUT_TSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "rank",
        "prophet",
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
        "marker_status",
        "gold_relation",
        "review_status",
        "sentence",
    ]

    with OUTPUT_TSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            delimiter="\t",
        )

        writer.writeheader()

        for row in candidates:
            writer.writerow(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column in columns
                }
            )

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in candidates:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        "Saved TSV:",
        OUTPUT_TSV,
    )

    print(
        "Saved JSONL:",
        OUTPUT_JSONL,
    )

    print()
    print(
        "Candidates by prophet:"
    )

    for prophet in [
        "nuh",
        "musa",
        "yunus",
        "ibrahim",
        "salih",
    ]:
        prophet_rows = [
            row
            for row in candidates
            if row["prophet"]
            == prophet
        ]

        predicted_relation = sum(
            row[
                "current_prediction"
            ]
            == "relation"
            for row in prophet_rows
        )

        print(
            f"{prophet:8s}: "
            f"{len(prophet_rows):4d} pairs | "
            f"{predicted_relation:4d} >= 0.5"
        )

    print()
    print(
        "=" * 120
    )
    print(
        "TOP 30 CANDIDATES"
    )
    print(
        "=" * 120
    )

    for row in candidates[:30]:
        print(
            f'{row["rank"]:3d} | '
            f'{row["prophet"]:8s} | '
            f'{row["surah"]}:{row["ayah"]:03d} | '
            f'{row["existence_probability"]:.6f} | '
            f'{row["subject"]} -> '
            f'{row["object"]} | '
            f'{row["marker_status"]}'
        )


if __name__ == "__main__":
    main()
