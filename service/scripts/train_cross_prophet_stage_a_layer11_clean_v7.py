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


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(
    "/u40/seroura/muslims_ml_project/"
    "dialect-aware-QA-chat-system-main-backup"
)

SERVICE_ROOT = PROJECT_ROOT / "service"

# READ-ONLY research repository.
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
    / "stage_a_train_cross_prophet_clean_v7.jsonl"
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
    / "cross_prophet_stage_a_layer11_clean_v7"
)

FOLD_DIRS = [
    RE_ROOT
    / "runs"
    / (
        f"arabertv02_marker_fgm_neraux015_twohead_oof_v1_"
        f"fold{fold}_seed42"
    )
    / "best_model"
    for fold in range(5)
]


# =====================================================================
# Experiment configuration
# =====================================================================

SEED = 42

MAX_LENGTH = 384
BATCH_SIZE = 8

# Deliberately conservative because we only have 40 training examples.
EPOCHS = 8

ENCODER_LR = 1e-5
EXISTENCE_HEAD_LR = 2e-4

WEIGHT_DECAY = 0.01

# Final production threshold remains unchanged.
EXISTENCE_THRESHOLD = 0.5


# =====================================================================
# General utilities
# =====================================================================

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
    return (
        0
        if row["relation"] == "no_relation"
        else 1
    )


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
# Original model loading
# =====================================================================

def build_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    label_to_id, id_to_label = (
        load_label_metadata(
            model_dir
        )
    )

    subject_marker_id = (
        tokenizer.convert_tokens_to_ids(
            "[SUBJ]"
        )
    )

    object_marker_id = (
        tokenizer.convert_tokens_to_ids(
            "[OBJ]"
        )
    )

    model = MultiTaskRelationModel(
        model_name=str(model_dir),
        num_relation_labels=len(
            label_to_id
        ),
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

    model.to(device)

    return model, tokenizer


# =====================================================================
# Dataset
# =====================================================================

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
                    "label": binary_label(
                        row
                    ),
                    "marker_status": (
                        marker_status
                    ),
                    "encoded": encoded,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(
        self,
        index,
    ):
        return self.examples[index]


class ExistenceCollator:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        examples,
    ):
        encoded = [
            example["encoded"]
            for example in examples
        ]

        batch = self.tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
        )

        batch["existence_labels"] = (
            torch.tensor(
                [
                    example["label"]
                    for example in examples
                ],
                dtype=torch.long,
            )
        )

        batch["example_ids"] = [
            example["id"]
            for example in examples
        ]

        return batch


# =====================================================================
# Trainable parameter policy
# =====================================================================

def configure_trainable_parameters(
    model,
):
    # Freeze absolutely everything first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze ONLY final BERT transformer layer.
    for parameter in (
        model
        .re_model
        .bert
        .encoder
        .layer[11]
        .parameters()
    ):
        parameter.requires_grad = True

    # Unfreeze ONLY Stage-A existence classifier.
    for parameter in (
        model
        .existence_classifier
        .parameters()
    ):
        parameter.requires_grad = True

    encoder_parameters = list(
        model
        .re_model
        .bert
        .encoder
        .layer[11]
        .parameters()
    )

    existence_parameters = list(
        model
        .existence_classifier
        .parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    total_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    return (
        encoder_parameters,
        existence_parameters,
        trainable_count,
        total_count,
    )


def verify_trainable_policy(
    model,
):
    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    illegal = [
        name
        for name in trainable_names
        if not (
            name.startswith(
                "re_model.bert.encoder.layer.11."
            )
            or name.startswith(
                "existence_classifier."
            )
        )
    ]

    if illegal:
        raise RuntimeError(
            "Unexpected trainable parameters:\n"
            + "\n".join(illegal)
        )

    print(
        "Trainable parameter groups verified."
    )

    print(
        "Trainable tensors:",
        len(trainable_names),
    )

    for name in trainable_names:
        print("  ", name)


# =====================================================================
# Forward path
# =====================================================================

def get_existence_logits(
    model,
    batch,
):
    model_inputs = {
        key: value
        for key, value
        in batch.items()
        if key not in {
            "existence_labels",
            "example_ids",
        }
    }

    # Use BertModel directly.
    #
    # This avoids computing the frozen 40-way relation classifier because
    # Stage A adaptation only needs the contextual [CLS] representation.
    bert_outputs = (
        model.re_model.bert(
            **model_inputs,
            return_dict=True,
        )
    )

    cls_representation = (
        bert_outputs.last_hidden_state[
            :,
            0,
            :,
        ]
    )

    existence_logits = (
        model.existence_classifier(
            model.existence_dropout(
                cls_representation
            )
        )
    )

    return existence_logits


# =====================================================================
# Evaluation
# =====================================================================

@torch.no_grad()
def predict_probabilities(
    model,
    loader,
    device,
):
    model.eval()

    probabilities = []
    labels = []
    ids = []

    for batch in loader:
        example_ids = batch.pop(
            "example_ids"
        )

        existence_labels = (
            batch.pop(
                "existence_labels"
            )
        )

        batch = {
            key: value.to(device)
            for key, value
            in batch.items()
        }

        logits = get_existence_logits(
            model,
            batch,
        )

        relation_probabilities = (
            torch.softmax(
                logits.float(),
                dim=-1,
            )[:, 1]
        )

        probabilities.extend(
            relation_probabilities
            .cpu()
            .tolist()
        )

        labels.extend(
            existence_labels.tolist()
        )

        ids.extend(
            example_ids
        )

    return (
        ids,
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
    )


def calculate_metrics(
    labels,
    probabilities,
):
    predictions = (
        probabilities
        >= EXISTENCE_THRESHOLD
    ).astype(
        np.int64
    )

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

    positive_total = tp + fn
    negative_total = tn + fp

    relation_recall = (
        tp / positive_total
        if positive_total
        else 0.0
    )

    no_relation_recall = (
        tn / negative_total
        if negative_total
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(labels)
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
        "relation_recall": (
            relation_recall
        ),
        "no_relation_recall": (
            no_relation_recall
        ),
        "balanced_accuracy": (
            balanced_accuracy
        ),
    }


def print_metrics(
    title,
    metrics,
):
    print()
    print(title)
    print("-" * 70)

    print(
        "TP:",
        metrics["tp"],
    )

    print(
        "TN:",
        metrics["tn"],
    )

    print(
        "FP:",
        metrics["fp"],
    )

    print(
        "FN:",
        metrics["fn"],
    )

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


# =====================================================================
# Training
# =====================================================================

def train_fold(
    fold,
    model,
    train_loader,
    device,
    class_weights,
):
    (
        encoder_parameters,
        existence_parameters,
        trainable_count,
        total_count,
    ) = configure_trainable_parameters(
        model
    )

    verify_trainable_policy(
        model
    )

    print()
    print(
        f"Fold {fold}: "
        f"{trainable_count:,} / "
        f"{total_count:,} "
        "parameters trainable"
    )

    optimizer = AdamW(
        [
            {
                "params": (
                    encoder_parameters
                ),
                "lr": ENCODER_LR,
                "weight_decay": (
                    WEIGHT_DECAY
                ),
            },
            {
                "params": (
                    existence_parameters
                ),
                "lr": (
                    EXISTENCE_HEAD_LR
                ),
                "weight_decay": (
                    WEIGHT_DECAY
                ),
            },
        ]
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        # Enables dropout during adaptation.
        model.train()

        running_losses = []

        for batch in train_loader:
            batch.pop(
                "example_ids"
            )

            labels = (
                batch.pop(
                    "existence_labels"
                )
                .to(device)
            )

            batch = {
                key: value.to(device)
                for key, value
                in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = (
                get_existence_logits(
                    model,
                    batch,
                )
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter
                    in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=1.0,
            )

            optimizer.step()

            running_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        mean_loss = float(
            np.mean(
                running_losses
            )
        )

        print(
            f"  epoch={epoch:02d} "
            f"loss={mean_loss:.6f}"
        )


# =====================================================================
# Save adaptation delta only
# =====================================================================

def save_adaptation_delta(
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
        / "stage_a_layer11_delta.pt"
    )

    torch.save(
        {
            "fold": fold,

            "encoder_layer_11": (
                model
                .re_model
                .bert
                .encoder
                .layer[11]
                .state_dict()
            ),

            "existence_classifier": (
                model
                .existence_classifier
                .state_dict()
            ),

            "source_checkpoint": str(
                FOLD_DIRS[fold]
            ),

            "training_data": str(
                TRAIN_PATH
            ),

            "training_method": (
                "final_encoder_layer_11_"
                "plus_existence_classifier"
            ),

            "epochs": EPOCHS,

            "encoder_lr": (
                ENCODER_LR
            ),

            "existence_head_lr": (
                EXISTENCE_HEAD_LR
            ),

            "weight_decay": (
                WEIGHT_DECAY
            ),

            "threshold": (
                EXISTENCE_THRESHOLD
            ),
        },
        output_path,
    )

    print()
    print(
        "Saved adaptation delta:",
        output_path,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Training data:",
        TRAIN_PATH,
    )

    print(
        "Held-out evaluation:",
        EVAL_PATH,
    )

    print(
        "Threshold:",
        EXISTENCE_THRESHOLD,
    )

    print(
        "Encoder LR:",
        ENCODER_LR,
    )

    print(
        "Existence-head LR:",
        EXISTENCE_HEAD_LR,
    )

    train_rows = load_jsonl(
        TRAIN_PATH
    )

    eval_rows = load_jsonl(
        EVAL_PATH
    )

    # ---------------------------------------------------------------
    # Verify there is no ayah leakage.
    # ---------------------------------------------------------------

    train_ayahs = {
        (
            int(row["surah"]),
            int(row["ayah"]),
        )
        for row in train_rows
    }

    eval_ayahs = {
        (
            int(row["surah"]),
            int(row["ayah"]),
        )
        for row in eval_rows
    }

    overlap = (
        train_ayahs
        & eval_ayahs
    )

    print(
        "Train/eval ayah overlap:",
        overlap,
    )

    if overlap:
        raise RuntimeError(
            "Training/evaluation leakage detected."
        )

    train_labels = np.asarray(
        [
            binary_label(row)
            for row in train_rows
        ],
        dtype=np.int64,
    )

    negative_count = int(
        np.sum(
            train_labels == 0
        )
    )

    positive_count = int(
        np.sum(
            train_labels == 1
        )
    )

    print()
    print(
        "Training rows:",
        len(train_rows),
    )

    print(
        "Relation:",
        positive_count,
    )

    print(
        "No relation:",
        negative_count,
    )

    # Balanced CE because this adaptation set deliberately contains
    # many hard-negative examples.
    total = len(
        train_rows
    )

    class_weights = torch.tensor(
        [
            total
            / (
                2.0
                * negative_count
            ),

            total
            / (
                2.0
                * positive_count
            ),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Class weights:",
        class_weights
        .detach()
        .cpu()
        .tolist(),
    )

    baseline_fold_probabilities = []
    adapted_fold_probabilities = []

    reference_ids = None
    reference_labels = None

    # ---------------------------------------------------------------
    # Five original Stage-A folds
    # ---------------------------------------------------------------

    for fold, model_dir in enumerate(
        FOLD_DIRS
    ):
        print()
        print("=" * 80)
        print(
            f"FOLD {fold}"
        )
        print("=" * 80)

        print(
            "Source checkpoint:",
            model_dir,
        )

        if not model_dir.exists():
            raise FileNotFoundError(
                model_dir
            )

        # Reset seed per fold for reproducibility.
        set_seed(
            SEED + fold
        )

        model, tokenizer = (
            build_model(
                model_dir,
                device,
            )
        )

        train_dataset = (
            ExistenceDataset(
                train_rows,
                tokenizer,
            )
        )

        eval_dataset = (
            ExistenceDataset(
                eval_rows,
                tokenizer,
            )
        )

        generator = (
            torch.Generator()
        )

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

        # -----------------------------------------------------------
        # Original fold before adaptation
        # -----------------------------------------------------------

        (
            eval_ids,
            eval_labels,
            baseline_probs,
        ) = predict_probabilities(
            model,
            eval_loader,
            device,
        )

        if reference_ids is None:
            reference_ids = (
                eval_ids
            )

            reference_labels = (
                eval_labels
            )

        else:
            assert (
                reference_ids
                == eval_ids
            )

            assert np.array_equal(
                reference_labels,
                eval_labels,
            )

        baseline_fold_probabilities.append(
            baseline_probs
        )

        # -----------------------------------------------------------
        # Limited Stage-A adaptation
        # -----------------------------------------------------------

        train_fold(
            fold,
            model,
            train_loader,
            device,
            class_weights,
        )

        # -----------------------------------------------------------
        # Prediction after adaptation
        # -----------------------------------------------------------

        (
            _,
            _,
            adapted_probs,
        ) = predict_probabilities(
            model,
            eval_loader,
            device,
        )

        adapted_fold_probabilities.append(
            adapted_probs
        )

        save_adaptation_delta(
            fold,
            model,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Five-fold ensemble
    # ---------------------------------------------------------------

    baseline_ensemble = np.mean(
        np.stack(
            baseline_fold_probabilities,
            axis=0,
        ),
        axis=0,
    )

    adapted_ensemble = np.mean(
        np.stack(
            adapted_fold_probabilities,
            axis=0,
        ),
        axis=0,
    )

    baseline_metrics = (
        calculate_metrics(
            reference_labels,
            baseline_ensemble,
        )
    )

    adapted_metrics = (
        calculate_metrics(
            reference_labels,
            adapted_ensemble,
        )
    )

    print_metrics(
        "ORIGINAL STAGE-A ENSEMBLE",
        baseline_metrics,
    )

    print_metrics(
        "LAYER-11 ADAPTED STAGE-A ENSEMBLE",
        adapted_metrics,
    )

    print()
    print("=" * 105)
    print(
        "Per-example comparison"
    )
    print("=" * 105)

    for (
        example_id,
        gold,
        before,
        after,
    ) in zip(
        reference_ids,
        reference_labels,
        baseline_ensemble,
        adapted_ensemble,
    ):
        print(
            f"{example_id:18s} "
            f"gold={gold} "
            f"before={before:.6f} "
            f"after={after:.6f} "
            f"delta={after-before:+.6f}"
        )

    # ---------------------------------------------------------------
    # Save diagnostic summary
    # ---------------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "experiment": (
            "cross_prophet_stage_a_layer11_clean_v7"
        ),

        "training_rows": (
            len(train_rows)
        ),

        "evaluation_rows": (
            len(eval_rows)
        ),

        "train_relation": (
            positive_count
        ),

        "train_no_relation": (
            negative_count
        ),

        "epochs": EPOCHS,

        "encoder_lr": (
            ENCODER_LR
        ),

        "existence_head_lr": (
            EXISTENCE_HEAD_LR
        ),

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "threshold": (
            EXISTENCE_THRESHOLD
        ),

        "baseline_metrics": (
            baseline_metrics
        ),

        "adapted_metrics": (
            adapted_metrics
        ),

        "examples": [
            {
                "id": example_id,
                "gold_exists": int(
                    gold
                ),
                "baseline_probability": (
                    float(before)
                ),
                "adapted_probability": (
                    float(after)
                ),
            }
            for (
                example_id,
                gold,
                before,
                after,
            ) in zip(
                reference_ids,
                reference_labels,
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
    print(
        "Saved evaluation summary:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()
