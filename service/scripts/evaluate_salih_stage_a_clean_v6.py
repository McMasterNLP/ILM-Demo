#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


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


EVAL_PATH = (
    SERVICE_ROOT
    / "data/re_adaptation/"
    "salih_stage_a_eval_v1.jsonl"
)

ADAPTATION_ROOT = (
    SERVICE_ROOT
    / "models/re_adaptation/"
    "cross_prophet_stage_a_layer11_clean_v6"
)

OUTPUT_PATH = (
    SERVICE_ROOT
    / "models/re_adaptation/"
    "cross_prophet_stage_a_layer11_clean_v6/"
    "salih_heldout_evaluation_v1.json"
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

MAX_LENGTH = 384
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def binary_label(row):
    return (
        0
        if row["relation"] == "no_relation"
        else 1
    )


def load_metadata(model_dir):
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


def load_original_fold(fold):
    model_dir = FOLD_DIRS[fold]

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    label_to_id, id_to_label = (
        load_metadata(model_dir)
    )

    model = MultiTaskRelationModel(
        model_name=str(model_dir),
        num_relation_labels=len(label_to_id),
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
        model_dir / "ner_auxiliary_heads.pt",
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

    model.to(DEVICE)
    model.eval()

    return model, tokenizer


def apply_adaptation_delta(
    model,
    fold,
):
    delta_path = (
        ADAPTATION_ROOT
        / f"fold{fold}"
        / "stage_a_layer11_delta.pt"
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


@torch.no_grad()
def score_row(
    model,
    tokenizer,
    row,
):
    marked_sentence, marker_status = (
        insert_markers(
            row["sentence"],
            row["subject"],
            row["object"],
        )
    )

    if marker_status != "marked":
        raise RuntimeError(
            f'{row["id"]}: unexpected marker status '
            f'{marker_status}'
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

    bert_outputs = model.re_model.bert(
        **encoded,
        return_dict=True,
    )

    cls_representation = (
        bert_outputs.last_hidden_state[
            :,
            0,
            :,
        ]
    )

    logits = model.existence_classifier(
        model.existence_dropout(
            cls_representation
        )
    )

    probability = torch.softmax(
        logits.float(),
        dim=-1,
    )[0, 1].item()

    return probability


def calculate_metrics(
    labels,
    probabilities,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int64)

    tp = int(np.sum(
        (labels == 1)
        & (predictions == 1)
    ))

    tn = int(np.sum(
        (labels == 0)
        & (predictions == 0)
    ))

    fp = int(np.sum(
        (labels == 0)
        & (predictions == 1)
    ))

    fn = int(np.sum(
        (labels == 1)
        & (predictions == 0)
    ))

    relation_recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    no_relation_recall = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(labels)
    )

    balanced_accuracy = (
        relation_recall
        + no_relation_recall
    ) / 2.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "relation_recall": relation_recall,
        "no_relation_recall": no_relation_recall,
        "balanced_accuracy": balanced_accuracy,
    }


def print_metrics(title, metrics):
    print()
    print(title)
    print("-" * 70)

    print("TP:", metrics["tp"])
    print("TN:", metrics["tn"])
    print("FP:", metrics["fp"])
    print("FN:", metrics["fn"])

    print(
        "Accuracy:           "
        f'{metrics["accuracy"]:.4f}'
    )

    print(
        "Relation recall:    "
        f'{metrics["relation_recall"]:.4f}'
    )

    print(
        "No-relation recall: "
        f'{metrics["no_relation_recall"]:.4f}'
    )

    print(
        "Balanced accuracy:  "
        f'{metrics["balanced_accuracy"]:.4f}'
    )


def main():
    rows = load_jsonl(
        EVAL_PATH
    )

    print("Device:", DEVICE)
    print("Evaluation rows:", len(rows))
    print("Threshold:", THRESHOLD)

    gold = [
        binary_label(row)
        for row in rows
    ]

    original_fold_scores = []
    adapted_fold_scores = []

    for fold in range(5):
        print(
            f"Evaluating fold {fold}..."
        )

        model, tokenizer = (
            load_original_fold(fold)
        )

        original_scores = [
            score_row(
                model,
                tokenizer,
                row,
            )
            for row in rows
        ]

        original_fold_scores.append(
            np.asarray(
                original_scores,
                dtype=np.float64,
            )
        )

        apply_adaptation_delta(
            model,
            fold,
        )

        adapted_scores = [
            score_row(
                model,
                tokenizer,
                row,
            )
            for row in rows
        ]

        adapted_fold_scores.append(
            np.asarray(
                adapted_scores,
                dtype=np.float64,
            )
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    original_ensemble = np.mean(
        np.stack(
            original_fold_scores,
            axis=0,
        ),
        axis=0,
    )

    adapted_ensemble = np.mean(
        np.stack(
            adapted_fold_scores,
            axis=0,
        ),
        axis=0,
    )

    original_metrics = calculate_metrics(
        gold,
        original_ensemble,
    )

    adapted_metrics = calculate_metrics(
        gold,
        adapted_ensemble,
    )

    print_metrics(
        "ORIGINAL STAGE-A ENSEMBLE",
        original_metrics,
    )

    print_metrics(
        "LAYER-11 ADAPTED STAGE-A ENSEMBLE",
        adapted_metrics,
    )

    print()
    print("=" * 125)
    print("PER-EXAMPLE COMPARISON")
    print("=" * 125)

    results = []

    for (
        row,
        gold_label,
        before,
        after,
    ) in zip(
        rows,
        gold,
        original_ensemble,
        adapted_ensemble,
    ):
        before_pred = (
            1 if before >= THRESHOLD else 0
        )

        after_pred = (
            1 if after >= THRESHOLD else 0
        )

        print(
            f'{row["id"]:34s} | '
            f'{row["prophet"]:8s} | '
            f'gold={gold_label} | '
            f'before={before:.6f} '
            f'pred={before_pred} | '
            f'after={after:.6f} '
            f'pred={after_pred} | '
            f'{row["subject"]} -> '
            f'{row["object"]}'
        )

        results.append(
            {
                "id": row["id"],
                "prophet": row["prophet"],
                "surah": row["surah"],
                "ayah": row["ayah"],
                "subject": row["subject"],
                "object": row["object"],
                "gold_relation": (
                    row["relation"]
                ),
                "gold_exists": (
                    gold_label
                ),
                "original_probability": (
                    float(before)
                ),
                "original_prediction": (
                    before_pred
                ),
                "adapted_probability": (
                    float(after)
                ),
                "adapted_prediction": (
                    after_pred
                ),
            }
        )

    print()
    print("=" * 80)
    print("BY PROPHET")
    print("=" * 80)

    by_prophet = {}

    for prophet in sorted(
        set(
            row["prophet"]
            for row in rows
        )
    ):
        indices = [
            index
            for index, row
            in enumerate(rows)
            if row["prophet"] == prophet
        ]

        prophet_gold = [
            gold[index]
            for index in indices
        ]

        prophet_original = [
            original_ensemble[index]
            for index in indices
        ]

        prophet_adapted = [
            adapted_ensemble[index]
            for index in indices
        ]

        original_p_metrics = (
            calculate_metrics(
                prophet_gold,
                prophet_original,
            )
        )

        adapted_p_metrics = (
            calculate_metrics(
                prophet_gold,
                prophet_adapted,
            )
        )

        by_prophet[prophet] = {
            "original": (
                original_p_metrics
            ),
            "adapted": (
                adapted_p_metrics
            ),
        }

        print()
        print(
            f"{prophet.upper()} "
            f"({len(indices)} examples)"
        )

        print(
            "  Original:"
            f' acc={original_p_metrics["accuracy"]:.3f}'
            f' rel_recall={original_p_metrics["relation_recall"]:.3f}'
            f' no_rel_recall={original_p_metrics["no_relation_recall"]:.3f}'
        )

        print(
            "  Adapted: "
            f' acc={adapted_p_metrics["accuracy"]:.3f}'
            f' rel_recall={adapted_p_metrics["relation_recall"]:.3f}'
            f' no_rel_recall={adapted_p_metrics["no_relation_recall"]:.3f}'
        )

    summary = {
        "evaluation_file": str(
            EVAL_PATH
        ),
        "threshold": THRESHOLD,
        "rows": len(rows),
        "original_metrics": (
            original_metrics
        ),
        "adapted_metrics": (
            adapted_metrics
        ),
        "by_prophet": (
            by_prophet
        ),
        "examples": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
