#!/usr/bin/env python3

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(
    "/u40/seroura/muslims_ml_project/"
    "dialect-aware-QA-chat-system-main-backup"
)

SERVICE_ROOT = PROJECT_ROOT / "service"

# READ-ONLY original research project.
RE_ROOT = Path(
    "/u40/seroura/knowledgegrapheval_re_2026"
)

sys.path.insert(0, str(RE_ROOT / "src"))

from train_marker_fgm_ner_aux_twohead_v1 import (
    MultiTaskRelationModel,
    insert_markers,
)


TRAIN_PATH = (
    SERVICE_ROOT
    / "data"
    / "re_adaptation"
    / "yusuf_stage_a_train_v1.jsonl"
)

EVAL_PATH = (
    SERVICE_ROOT
    / "data"
    / "re_adaptation"
    / "yusuf_stage_a_eval_v0.jsonl"
)

OUTPUT_ROOT = (
    SERVICE_ROOT
    / "models"
    / "re_adaptation"
    / "yusuf_stage_a_existence_v1"
)

FOLD_DIRS = [
    RE_ROOT
    / (
        "runs/"
        f"arabertv02_marker_fgm_neraux015_twohead_oof_v1_fold{fold}"
        "_seed42/best_model"
    )
    for fold in range(5)
]


# ---------------------------------------------------------------------
# Fixed diagnostic experiment configuration
# ---------------------------------------------------------------------

SEED = 42
MAX_LENGTH = 384
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01

# Final system threshold remains unchanged.
EXISTENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def binary_label(row):
    return 0 if row["relation"] == "no_relation" else 1


def load_label_metadata(model_dir):
    metadata_path = model_dir / "training_metadata.json"

    metadata = json.loads(
        metadata_path.read_text(
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


def build_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    label_to_id, id_to_label = (
        load_label_metadata(model_dir)
    )

    subject_marker_id = tokenizer.convert_tokens_to_ids(
        "[SUBJ]"
    )

    object_marker_id = tokenizer.convert_tokens_to_ids(
        "[OBJ]"
    )

    model = MultiTaskRelationModel(
        model_name=str(model_dir),
        num_relation_labels=len(label_to_id),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        subject_marker_id=subject_marker_id,
        object_marker_id=object_marker_id,
    )

    auxiliary_path = (
        model_dir
        / "ner_auxiliary_heads.pt"
    )

    auxiliary = torch.load(
        auxiliary_path,
        map_location="cpu",
        weights_only=False,
    )

    model.subject_type_classifier.load_state_dict(
        auxiliary["subject_type_classifier"]
    )

    model.object_type_classifier.load_state_dict(
        auxiliary["object_type_classifier"]
    )

    model.existence_classifier.load_state_dict(
        auxiliary["existence_classifier"]
    )

    model.to(device)

    return model, tokenizer


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class ExistenceDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
    ):
        self.examples = []

        for row in rows:
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
                padding=False,
            )

            self.examples.append(
                {
                    "id": row["id"],
                    "label": binary_label(row),
                    "marker_status": marker_status,
                    "encoded": encoded,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class ExistenceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        encoded = [
            example["encoded"]
            for example in examples
        ]

        batch = self.tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
        )

        batch["existence_labels"] = torch.tensor(
            [
                example["label"]
                for example in examples
            ],
            dtype=torch.long,
        )

        batch["example_ids"] = [
            example["id"]
            for example in examples
        ]

        return batch


# ---------------------------------------------------------------------
# Frozen encoder + trainable existence head
# ---------------------------------------------------------------------

def freeze_everything_except_existence(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in (
        model.existence_classifier.parameters()
    ):
        parameter.requires_grad = True

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return trainable, total


def existence_logits(
    model,
    batch,
    *,
    training,
):
    model.re_model.eval()

    if training:
        model.existence_dropout.train()
        model.existence_classifier.train()
    else:
        model.existence_dropout.eval()
        model.existence_classifier.eval()

    model_inputs = {
        key: value
        for key, value in batch.items()
        if key not in {
            "existence_labels",
            "example_ids",
        }
    }

    # Encoder is completely frozen.
    with torch.no_grad():
        outputs = model.re_model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        cls_representation = (
            outputs.hidden_states[-1][:, 0, :]
        )

    logits = model.existence_classifier(
        model.existence_dropout(
            cls_representation
        )
    )

    return logits


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_probabilities(
    model,
    loader,
    device,
):
    probabilities = []
    labels = []
    ids = []

    for batch in loader:
        example_ids = batch.pop(
            "example_ids"
        )

        existence_labels = batch[
            "existence_labels"
        ]

        batch = {
            key: value.to(device)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }

        logits = existence_logits(
            model,
            batch,
            training=False,
        )

        probs = torch.softmax(
            logits.float(),
            dim=-1,
        )[:, 1]

        probabilities.extend(
            probs.cpu().tolist()
        )

        labels.extend(
            existence_labels.tolist()
        )

        ids.extend(example_ids)

    return ids, np.asarray(labels), np.asarray(
        probabilities
    )


def metrics_from_probabilities(
    labels,
    probabilities,
):
    predictions = (
        probabilities
        >= EXISTENCE_THRESHOLD
    ).astype(np.int64)

    tp = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )

    relation_total = tp + fn
    no_relation_total = tn + fp

    relation_recall = (
        tp / relation_total
        if relation_total
        else 0.0
    )

    no_relation_recall = (
        tn / no_relation_total
        if no_relation_total
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(labels)
        if len(labels)
        else 0.0
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


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_fold(
    fold,
    model,
    train_loader,
    device,
    class_weights,
):
    trainable, total = (
        freeze_everything_except_existence(
            model
        )
    )

    print(
        f"Fold {fold}: trainable "
        f"{trainable:,} / {total:,} parameters"
    )

    optimizer = AdamW(
        model.existence_classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        losses = []

        for batch in train_loader:
            batch.pop("example_ids")

            labels = batch.pop(
                "existence_labels"
            ).to(device)

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = existence_logits(
                model,
                batch,
                training=True,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.existence_classifier.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            losses.append(
                float(loss.detach().cpu())
            )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"  epoch={epoch:02d} "
                f"loss={np.mean(losses):.6f}"
            )


def save_adapted_head(
    fold,
    model,
):
    fold_output = (
        OUTPUT_ROOT
        / f"fold{fold}"
    )

    fold_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fold_output
        / "existence_classifier.pt"
    )

    torch.save(
        {
            "fold": fold,
            "existence_classifier": (
                model.existence_classifier.state_dict()
            ),
            "source_checkpoint": str(
                FOLD_DIRS[fold]
            ),
            "training_data": str(
                TRAIN_PATH
            ),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "threshold": EXISTENCE_THRESHOLD,
            "trainable_component": (
                "existence_classifier_only"
            ),
        },
        output_path,
    )

    print(
        "Saved adapted head:",
        output_path,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Training:", TRAIN_PATH)
    print("Evaluation:", EVAL_PATH)
    print("Threshold:", EXISTENCE_THRESHOLD)

    train_rows = load_jsonl(
        TRAIN_PATH
    )

    eval_rows = load_jsonl(
        EVAL_PATH
    )

    train_labels = np.asarray(
        [
            binary_label(row)
            for row in train_rows
        ]
    )

    negative_count = int(
        np.sum(train_labels == 0)
    )

    positive_count = int(
        np.sum(train_labels == 1)
    )

    print()
    print("Training rows:", len(train_rows))
    print("No relation:", negative_count)
    print("Relation:", positive_count)

    # Balanced CE:
    # each class contributes approximately equally despite
    # the intentionally hard-negative-heavy adaptation pool.
    total = len(train_rows)

    class_weights = torch.tensor(
        [
            total / (2.0 * negative_count),
            total / (2.0 * positive_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Class weights:",
        class_weights.cpu().tolist(),
    )

    baseline_fold_probs = []
    adapted_fold_probs = []

    eval_labels_reference = None
    eval_ids_reference = None

    for fold, model_dir in enumerate(
        FOLD_DIRS
    ):
        print()
        print("=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)
        print("Source checkpoint:", model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                model_dir
            )

        model, tokenizer = build_model(
            model_dir,
            device,
        )

        train_dataset = ExistenceDataset(
            train_rows,
            tokenizer,
        )

        eval_dataset = ExistenceDataset(
            eval_rows,
            tokenizer,
        )

        generator = torch.Generator()
        generator.manual_seed(
            SEED + fold
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            collate_fn=ExistenceCollator(
                tokenizer
            ),
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=ExistenceCollator(
                tokenizer
            ),
        )

        # -------------------------------------------------------------
        # Baseline fold prediction BEFORE adaptation
        # -------------------------------------------------------------

        eval_ids, eval_labels, baseline_probs = (
            predict_probabilities(
                model,
                eval_loader,
                device,
            )
        )

        if eval_labels_reference is None:
            eval_labels_reference = (
                eval_labels
            )
            eval_ids_reference = eval_ids
        else:
            assert np.array_equal(
                eval_labels_reference,
                eval_labels,
            )

            assert (
                eval_ids_reference
                == eval_ids
            )

        baseline_fold_probs.append(
            baseline_probs
        )

        # -------------------------------------------------------------
        # Adapt existence head only
        # -------------------------------------------------------------

        train_fold(
            fold,
            model,
            train_loader,
            device,
            class_weights,
        )

        # -------------------------------------------------------------
        # Adapted fold prediction
        # -------------------------------------------------------------

        _, _, adapted_probs = (
            predict_probabilities(
                model,
                eval_loader,
                device,
            )
        )

        adapted_fold_probs.append(
            adapted_probs
        )

        save_adapted_head(
            fold,
            model,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Five-fold ensemble comparison
    # -----------------------------------------------------------------

    baseline_ensemble = np.mean(
        np.stack(
            baseline_fold_probs,
            axis=0,
        ),
        axis=0,
    )

    adapted_ensemble = np.mean(
        np.stack(
            adapted_fold_probs,
            axis=0,
        ),
        axis=0,
    )

    baseline_metrics = (
        metrics_from_probabilities(
            eval_labels_reference,
            baseline_ensemble,
        )
    )

    adapted_metrics = (
        metrics_from_probabilities(
            eval_labels_reference,
            adapted_ensemble,
        )
    )

    print_metrics(
        "ORIGINAL STAGE-A ENSEMBLE",
        baseline_metrics,
    )

    print_metrics(
        "ADAPTED EXISTENCE-HEAD ENSEMBLE",
        adapted_metrics,
    )

    print()
    print("=" * 100)
    print("Per-example comparison")
    print("=" * 100)

    for (
        example_id,
        label,
        before,
        after,
    ) in zip(
        eval_ids_reference,
        eval_labels_reference,
        baseline_ensemble,
        adapted_ensemble,
    ):
        print(
            f"{example_id:18s} "
            f"gold={label} "
            f"before={before:.6f} "
            f"after={after:.6f} "
            f"delta={after-before:+.6f}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "experiment": (
            "yusuf_stage_a_existence_head_only_v1"
        ),
        "training_rows": len(
            train_rows
        ),
        "evaluation_rows": len(
            eval_rows
        ),
        "train_positive": positive_count,
        "train_no_relation": negative_count,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "threshold": EXISTENCE_THRESHOLD,
        "baseline_metrics": baseline_metrics,
        "adapted_metrics": adapted_metrics,
        "examples": [
            {
                "id": example_id,
                "gold_exists": int(label),
                "baseline_probability": float(
                    before
                ),
                "adapted_probability": float(
                    after
                ),
            }
            for (
                example_id,
                label,
                before,
                after,
            ) in zip(
                eval_ids_reference,
                eval_labels_reference,
                baseline_ensemble,
                adapted_ensemble,
            )
        ],
    }

    summary_path = (
        OUTPUT_ROOT
        / "evaluation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved evaluation summary:")
    print(summary_path)


if __name__ == "__main__":
    main()
