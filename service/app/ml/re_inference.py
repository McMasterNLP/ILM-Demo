from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)


# ============================================================================
# Original KnowledgeGraphEval RE project
# READ-ONLY
# ============================================================================

RE_PROJECT_ROOT = Path(
    "/u40/seroura/knowledgegrapheval_re_2026"
)

RE_SRC = RE_PROJECT_ROOT / "src"

if str(RE_SRC) not in sys.path:
    sys.path.insert(0, str(RE_SRC))


from train_marker_fgm_ner_aux_twohead_v1 import (
    MultiTaskRelationModel,
    insert_markers,
)

from predict_nli_fgmner_v2_ensemble import (
    HiddenNLIDataset,
    HiddenNLICollator,
    stable_softmax,
)


# ============================================================================
# Runtime configuration
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EXISTENCE_THRESHOLD = 0.5
MAX_LENGTH = 384
TOP_K = 5

ARABERT_WEIGHT = 0.55
NLI_WEIGHT = 0.45
NLI_TEMPERATURE = 0.75
NLI_BATCH_SIZE = 64


TWOHEAD_MODEL_DIRS = [
    RE_PROJECT_ROOT
    / "runs"
    / f"arabertv02_marker_fgm_neraux015_twohead_oof_v1_fold{fold}_seed42"
    / "best_model"
    for fold in range(5)
]


POSITIVE_MODEL_DIRS = [
    RE_PROJECT_ROOT
    / "runs"
    / f"arabertv02_marker_fgm_neraux015_v1_fold{fold}_seed42"
    / "best_model"
    for fold in range(5)
]


NLI_MODEL_DIRS = [
    RE_PROJECT_ROOT
    / "runs"
    / f"arabert_nli_reranker_fgmner_v2_fold{fold}_seed42"
    / "best_model"
    for fold in range(5)
]


NLI_TEMPLATE_PATH = (
    RE_PROJECT_ROOT
    / "reports"
    / "nli_templates.json"
)


# V6 Prophet-story Stage-A adaptation.
#
# The original KnowledgeGraphEval checkpoints remain read-only.
# Each V6 delta replaces only:
#   1. final BERT encoder layer 11
#   2. relation-existence classifier
V6_ADAPTATION_ROOT = Path(
    "/u40/seroura/muslims_ml_project/"
    "dialect-aware-QA-chat-system-main-backup/"
    "service/models/re_adaptation/"
    "cross_prophet_stage_a_layer11_clean_v6"
)

V6_DELTA_PATHS = [
    V6_ADAPTATION_ROOT
    / f"fold{fold}"
    / "stage_a_layer11_delta.pt"
    for fold in range(5)
]


# ============================================================================
# Public result structures
# ============================================================================

@dataclass
class REPrediction:
    subject_text: str
    object_text: str
    relation_label: str
    relation_confidence: float | None
    existence_probability: float


@dataclass
class NLIRerankedCandidate:
    relation_label: str
    arabert_probability: float
    nli_probability: float
    blended_probability: float


# ============================================================================
# Shared helpers
# ============================================================================

def build_model_input(
    *,
    tokenizer,
    sentence: str,
    subject: str,
    object_: str,
):
    marked_sentence, marker_status = insert_markers(
        sentence,
        subject,
        object_,
    )

    pair_sequence = (
        f"[SUBJ] {subject} [/SUBJ] "
        f"[OBJ] {object_} [/OBJ]"
    )

    encoded = tokenizer(
        marked_sentence,
        pair_sequence,
        truncation="only_first",
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    return encoded, marker_status


def load_saved_tokenizer(model_dir: Path):
    """
    Load the exact saved tokenizer vocabulary without parsing a legacy
    tokenizer_config.json whose extra_special_tokens representation can be
    incompatible with newer Transformers versions.

    No vocabulary entries are added here. Existing BERT special tokens and
    the four trained RE marker tokens are bound to their saved vocabulary IDs.
    """
    tokenizer_json = model_dir / "tokenizer.json"

    if not tokenizer_json.exists():
        return AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
        )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
    )

    # These tokens already exist in the saved vocabulary.
    # Assigning them here does NOT add new vocabulary entries.
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.mask_token = "[MASK]"

    expected_markers = {
        "[SUBJ]": 64000,
        "[/SUBJ]": 64001,
        "[OBJ]": 64002,
        "[/OBJ]": 64003,
    }

    for token, expected_id in expected_markers.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)

        if actual_id != expected_id:
            raise RuntimeError(
                f"RE marker ID mismatch for {token}: "
                f"expected {expected_id}, found {actual_id}"
            )

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Saved RE tokenizer has no usable [PAD] token."
        )

    return tokenizer


def load_label_maps(model_dir: Path):
    metadata_path = (
        model_dir / "training_metadata.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    label_to_id = {
        label: int(index)
        for label, index
        in metadata["label_to_id"].items()
    }

    id_to_label = {
        int(index): label
        for index, label
        in metadata["id_to_label"].items()
    }

    return label_to_id, id_to_label


# ============================================================================
# Stage A: relation existence
# ============================================================================

class RelationExistenceEnsemble:

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for RE inference."
            )

        self.tokenizer = load_saved_tokenizer(
            TWOHEAD_MODEL_DIRS[0]
        )

        self.subject_marker_id = (
            self.tokenizer.convert_tokens_to_ids(
                "[SUBJ]"
            )
        )

        self.object_marker_id = (
            self.tokenizer.convert_tokens_to_ids(
                "[OBJ]"
            )
        )

        self.models = []

        for model_dir in TWOHEAD_MODEL_DIRS:
            label_to_id, id_to_label = (
                load_label_maps(model_dir)
            )

            model = MultiTaskRelationModel(
                model_name=str(model_dir),
                num_relation_labels=len(
                    label_to_id
                ),
                label_to_id=label_to_id,
                id_to_label=id_to_label,
                subject_marker_id=(
                    self.subject_marker_id
                ),
                object_marker_id=(
                    self.object_marker_id
                ),
            )

            auxiliary_heads = torch.load(
                model_dir
                / "ner_auxiliary_heads.pt",
                map_location="cpu",
            )

            model.subject_type_classifier.load_state_dict(
                auxiliary_heads[
                    "subject_type_classifier"
                ]
            )

            model.object_type_classifier.load_state_dict(
                auxiliary_heads[
                    "object_type_classifier"
                ]
            )

            model.existence_classifier.load_state_dict(
                auxiliary_heads[
                    "existence_classifier"
                ]
            )

            # ------------------------------------------------------------
            # Apply the selected V6 Prophet-story Stage-A adaptation.
            #
            # Only encoder layer 11 and the existence classifier were
            # trained during adaptation. Everything else remains from the
            # original source checkpoint.
            # ------------------------------------------------------------

            fold = len(self.models)
            delta_path = V6_DELTA_PATHS[fold]

            if not delta_path.exists():
                raise FileNotFoundError(
                    f"Missing V6 Stage-A delta: {delta_path}"
                )

            delta = torch.load(
                delta_path,
                map_location="cpu",
                weights_only=False,
            )

            if int(delta["fold"]) != fold:
                raise RuntimeError(
                    f"V6 delta fold mismatch: expected {fold}, "
                    f"found {delta['fold']} in {delta_path}"
                )

            expected_source = str(model_dir)

            if (
                str(delta["source_checkpoint"])
                != expected_source
            ):
                raise RuntimeError(
                    "V6 source checkpoint mismatch. "
                    f"Expected {expected_source!r}, "
                    f"found {delta['source_checkpoint']!r}"
                )

            model.re_model.bert.encoder.layer[
                11
            ].load_state_dict(
                delta["encoder_layer_11"]
            )

            model.existence_classifier.load_state_dict(
                delta["existence_classifier"]
            )

            model.to(DEVICE)
            model.eval()

            self.models.append(model)

    def predict_probability(
        self,
        *,
        sentence: str,
        subject: str,
        object_: str,
    ) -> float:

        encoded, _ = build_model_input(
            tokenizer=self.tokenizer,
            sentence=sentence,
            subject=subject,
            object_=object_,
        )

        fold_probabilities = []

        with torch.inference_mode():
            for model in self.models:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    # Match the exact V6 evaluation forward path:
                    # BertModel -> CLS -> existence dropout/classifier.
                    bert_outputs = (
                        model.re_model.bert(
                            **encoded,
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

                existence_probabilities = (
                    torch.softmax(
                        existence_logits.float(),
                        dim=-1,
                    )
                )

                fold_probabilities.append(
                    existence_probabilities[
                        0,
                        1,
                    ].item()
                )

        return float(
            sum(fold_probabilities)
            / len(fold_probabilities)
        )


# ============================================================================
# Stage B1: strong positive 40-way classifier
# ============================================================================

class PositiveRelationEnsemble:

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for RE inference."
            )

        self.tokenizer = load_saved_tokenizer(
            POSITIVE_MODEL_DIRS[0]
        )

        first_label_to_id, first_id_to_label = (
            load_label_maps(
                POSITIVE_MODEL_DIRS[0]
            )
        )

        self.label_to_id = first_label_to_id
        self.id_to_label = first_id_to_label

        self.models = []

        for model_dir in POSITIVE_MODEL_DIRS:

            label_to_id, id_to_label = (
                load_label_maps(model_dir)
            )

            if label_to_id != self.label_to_id:
                raise RuntimeError(
                    f"Label mapping mismatch: {model_dir}"
                )

            if id_to_label != self.id_to_label:
                raise RuntimeError(
                    f"ID mapping mismatch: {model_dir}"
                )

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(model_dir)
            )

            model.to(DEVICE)
            model.eval()

            self.models.append(model)

    def predict_probabilities(
        self,
        *,
        sentence: str,
        subject: str,
        object_: str,
    ) -> np.ndarray:

        encoded, _ = build_model_input(
            tokenizer=self.tokenizer,
            sentence=sentence,
            subject=subject,
            object_=object_,
        )

        fold_probabilities = []

        with torch.inference_mode():
            for model in self.models:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(**encoded)

                probabilities = torch.softmax(
                    outputs.logits.float(),
                    dim=-1,
                )

                fold_probabilities.append(
                    probabilities[
                        0
                    ].cpu().numpy()
                )

        return np.stack(
            fold_probabilities,
            axis=0,
        ).mean(axis=0)


# ============================================================================
# Stage B2: NLI reranking
# ============================================================================

class NLIReranker:

    def __init__(
        self,
        positive_ensemble: PositiveRelationEnsemble,
    ) -> None:

        self.positive_ensemble = positive_ensemble

        with NLI_TEMPLATE_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.templates = (
                json.load(file)["templates"]
            )

        self.models = []
        self.tokenizers = []

        for model_dir in NLI_MODEL_DIRS:

            tokenizer = (
                load_saved_tokenizer(
                    model_dir
                )
            )

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    model_dir
                )
            )

            model.to(DEVICE)
            model.eval()

            self.tokenizers.append(
                tokenizer
            )

            self.models.append(
                model
            )

    def _score_one_fold(
        self,
        *,
        model,
        tokenizer,
        dataset,
    ) -> np.ndarray:

        loader = DataLoader(
            dataset,
            batch_size=NLI_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=HiddenNLICollator(
                tokenizer=tokenizer,
                max_length=MAX_LENGTH,
            ),
        )

        score_sums = np.zeros(
            TOP_K,
            dtype=np.float64,
        )

        score_counts = np.zeros(
            TOP_K,
            dtype=np.int64,
        )

        with torch.inference_mode():

            for batch in loader:

                batch.pop("row_indices")

                candidate_indices = (
                    batch.pop(
                        "candidate_indices"
                    )
                    .numpy()
                )

                model_inputs = {
                    key: value.to(
                        DEVICE,
                        non_blocking=True,
                    )
                    for key, value
                    in batch.items()
                }

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        **model_inputs
                    )

                logits = (
                    outputs.logits
                    .squeeze(-1)
                    .float()
                    .cpu()
                    .numpy()
                )

                np.add.at(
                    score_sums,
                    candidate_indices,
                    logits,
                )

                np.add.at(
                    score_counts,
                    candidate_indices,
                    1,
                )

        if np.any(score_counts != 2):
            raise RuntimeError(
                "Each NLI candidate must receive "
                "exactly two template scores. "
                f"Counts: {score_counts.tolist()}"
            )

        return (
            score_sums
            / score_counts
        )

    def rerank(
        self,
        *,
        sentence: str,
        subject: str,
        object_: str,
    ) -> list[NLIRerankedCandidate]:

        arabert_probabilities = (
            self.positive_ensemble
            .predict_probabilities(
                sentence=sentence,
                subject=subject,
                object_=object_,
            )
        )

        top5_ids = np.argsort(
            -arabert_probabilities
        )[:TOP_K]

        top5_probabilities = (
            arabert_probabilities[
                top5_ids
            ]
            .astype(np.float64)
        )

        top5_probabilities = (
            top5_probabilities
            / top5_probabilities.sum()
        )

        candidate_relations = [
            self.positive_ensemble
            .id_to_label[
                int(label_id)
            ]
            for label_id
            in top5_ids
        ]

        test_rows = [
            {
                "sentence": sentence,
                "subject": subject,
                "object": object_,
            }
        ]

        dataset = HiddenNLIDataset(
            test_rows=test_rows,
            candidate_relations=[
                candidate_relations
            ],
            templates=self.templates,
        )

        all_fold_scores = []

        for model, tokenizer in zip(
            self.models,
            self.tokenizers,
        ):

            fold_scores = (
                self._score_one_fold(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                )
            )

            all_fold_scores.append(
                fold_scores
            )

        nli_ensemble_scores = (
            np.stack(
                all_fold_scores,
                axis=0,
            )
            .mean(axis=0)
        )

        nli_probabilities = (
            stable_softmax(
                nli_ensemble_scores[
                    None,
                    :
                ],
                temperature=(
                    NLI_TEMPERATURE
                ),
            )[0]
        )

        blended_probabilities = (
            ARABERT_WEIGHT
            * top5_probabilities
            + NLI_WEIGHT
            * nli_probabilities
        )

        results = [
            NLIRerankedCandidate(
                relation_label=relation,
                arabert_probability=float(
                    arabert_probability
                ),
                nli_probability=float(
                    nli_probability
                ),
                blended_probability=float(
                    blended_probability
                ),
            )
            for (
                relation,
                arabert_probability,
                nli_probability,
                blended_probability,
            )
            in zip(
                candidate_relations,
                top5_probabilities,
                nli_probabilities,
                blended_probabilities,
            )
        ]

        results.sort(
            key=lambda item:
                item.blended_probability,
            reverse=True,
        )

        return results


# ============================================================================
# Final public RE service
#
# For now:
# existence gate -> positive classifier -> NLI blend
#
# Schema adjustment will be added when NER entity types are available.
# ============================================================================

class RelationExtractionService:

    def __init__(self) -> None:
        self.existence = (
            RelationExistenceEnsemble()
        )

        self.positive = (
            PositiveRelationEnsemble()
        )

        self.nli = NLIReranker(
            self.positive
        )

    def predict(
        self,
        *,
        sentence: str,
        subject: str,
        object_: str,
    ) -> REPrediction:

        existence_probability = (
            self.existence
            .predict_probability(
                sentence=sentence,
                subject=subject,
                object_=object_,
            )
        )

        if (
            existence_probability
            < EXISTENCE_THRESHOLD
        ):
            return REPrediction(
                subject_text=subject,
                object_text=object_,
                relation_label="no_relation",
                relation_confidence=None,
                existence_probability=(
                    existence_probability
                ),
            )

        ranked_candidates = (
            self.nli.rerank(
                sentence=sentence,
                subject=subject,
                object_=object_,
            )
        )

        best = ranked_candidates[0]

        return REPrediction(
            subject_text=subject,
            object_text=object_,
            relation_label=(
                best.relation_label
            ),
            relation_confidence=(
                best.blended_probability
            ),
            existence_probability=(
                existence_probability
            ),
        )
