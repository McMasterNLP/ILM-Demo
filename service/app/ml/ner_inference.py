from __future__ import annotations

import importlib.util
import pickle
import re
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch


# =============================================================================
# Paths
# =============================================================================

NER_ROOT = Path("/u40/seroura/adaptner_tapt")
NER_SCRIPTS = NER_ROOT / "scripts"

if str(NER_ROOT) not in sys.path:
    sys.path.insert(0, str(NER_ROOT))

if str(NER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(NER_SCRIPTS))


# Primary token experts
ARABERT_MODEL_DIR = (
    NER_ROOT
    / "runs/zero_konooz_arabert_entity_swap_v4_10pct_ckpt2_lr5e6_seed42_v1"
)

CAMELBERT_MODEL_DIR = (
    NER_ROOT
    / "runs/camelbert_mix_base_entity_swap_v4_10pct_from_ckpt2_lr5e6"
)

MARBERTV2_MODEL_DIR = (
    NER_ROOT
    / "runs/marbertv2_base_entity_swap_v4_10pct_from_ckpt7_lr5e6"
)

# Span expert
SPAN_CHECKPOINT = (
    NER_ROOT
    / "runs/zero_konooz_span_arabertv2_e7_e8_v1/best.pt"
)

# XLM-R complementary expert
XLMR_MODEL_DIR = NER_ROOT / "models/xlm_roberta_base"

XLMR_CHECKPOINT = (
    NER_ROOT
    / "runs/xlmr_base_wojood_saveevery_lr2e5_seed42_v1"
    / "checkpoints/checkpoint_2.pt"
)

# This file was produced by the exact checkpoint-2 CV-18 evaluation.
XLMR_TAG_VOCAB = (
    NER_ROOT
    / "analysis/cv18_zero_shot/xlmr_base_ckpt2_devonly_v1"
    / "tag_vocab.pkl"
)


# =============================================================================
# Entity inventory
# =============================================================================

ENTITY_ORDER = [
    "CARDINAL",
    "CURR",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "OCC",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERS",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "UNIT",
    "WEBSITE",
]

MODEL_ORDER = [
    "AraBERT",
    "CAMeLBERT",
    "MARBERTv2",
    "LIPN",
]

MODEL_FAMILY = {
    "AraBERT": "arabert_family",
    "CAMeLBERT": "camelbert_family",
    "MARBERTv2": "marbert_family",
    "LIPN": "span_family",
    "XLMR": "xlmr_family",
}


# =============================================================================
# Public output
# =============================================================================

@dataclass(frozen=True)
class NEREntity:
    text: str
    entity_type: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    supporters: Tuple[str, ...]


# Internal span:
# (entity_type, token_start_inclusive, token_end_exclusive)
Span = Tuple[str, int, int]


# =============================================================================
# Dynamic import helpers
# =============================================================================

def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =============================================================================
# BIO helpers
# =============================================================================

def _tags_to_spans(
    tags: Sequence[Sequence[str]],
) -> Set[Span]:
    """
    Convert the 21 BIO columns for one sentence into exact nested spans.
    """
    spans: Set[Span] = set()

    if not tags:
        return spans

    for type_index, entity_type in enumerate(ENTITY_ORDER):
        start = None

        for token_index in range(len(tags) + 1):
            if token_index < len(tags):
                tag = tags[token_index][type_index]
            else:
                tag = "O"

            expected_b = f"B-{entity_type}"
            expected_i = f"I-{entity_type}"

            if tag == expected_b:
                if start is not None:
                    spans.add(
                        (
                            entity_type,
                            start,
                            token_index,
                        )
                    )

                start = token_index

            elif tag == expected_i:
                if start is None:
                    # Defensive BIO repair:
                    # an orphan I-X starts a new X span.
                    start = token_index

            else:
                if start is not None:
                    spans.add(
                        (
                            entity_type,
                            start,
                            token_index,
                        )
                    )
                    start = None

    return spans


def _prediction_lines_to_tags(
    lines: Sequence[str],
) -> List[List[str]]:
    tags: List[List[str]] = []

    for line in lines:
        columns = line.split()

        if len(columns) != 22:
            raise ValueError(
                "Expected token plus 21 NER columns, "
                f"found {len(columns)} columns: {line!r}"
            )

        tags.append(columns[1:])

    return tags


# =============================================================================
# Ensemble helpers
# =============================================================================

def _overlaps(first: Span, second: Span) -> bool:
    return (
        first[0] == second[0]
        and max(first[1], second[1])
        < min(first[2], second[2])
    )


def _span_length(span: Span) -> int:
    return span[2] - span[1]


def _build_components(spans: Set[Span]) -> List[Set[Span]]:
    """
    Build connected overlap components independently for each entity type.
    """
    remaining = set(spans)
    components: List[Set[Span]] = []

    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = [seed]

        while queue:
            current = queue.pop()

            neighbours = {
                candidate
                for candidate in remaining
                if _overlaps(current, candidate)
            }

            if neighbours:
                remaining.difference_update(neighbours)
                component.update(neighbours)
                queue.extend(neighbours)

        components.append(component)

    return components


def _get_supporters(
    span: Span,
    model_spans: Dict[str, Set[Span]],
    model_order: Sequence[str],
) -> Tuple[str, ...]:
    return tuple(
        model_name
        for model_name in model_order
        if span in model_spans[model_name]
    )


def _build_primary_consensus(
    model_spans: Dict[str, Set[Span]],
) -> Set[Span]:
    """
    Exact final four-primary-expert policy:

    - AraBERT
    - CAMeLBERT
    - MARBERTv2
    - LIPN/span

    Require at least two exact votes.

    Within an overlapping component:
      1. highest exact vote count
      2. shorter span
      3. earlier start
      4. earlier end
    """
    union = set().union(
        *[
            model_spans[model_name]
            for model_name in MODEL_ORDER
        ]
    )

    components = _build_components(union)
    output: Set[Span] = set()

    for component in components:
        candidates = []

        for span in component:
            supporters = _get_supporters(
                span,
                model_spans,
                MODEL_ORDER,
            )

            vote_count = len(supporters)

            if vote_count < 2:
                continue

            candidates.append(
                (
                    span,
                    vote_count,
                )
            )

        if not candidates:
            continue

        selected_span, _ = max(
            candidates,
            key=lambda item: (
                item[1],
                -_span_length(item[0]),
                -item[0][1],
                -item[0][2],
            ),
        )

        output.add(selected_span)

    return output


def _build_family_aware_with_xlmr(
    model_spans: Dict[str, Set[Span]],
) -> Set[Span]:
    """
    Reproduce the family-aware candidate construction used before selecting
    safe XLM-R additions.
    """
    model_order = MODEL_ORDER + ["XLMR"]

    union = set().union(
        *[
            model_spans[model_name]
            for model_name in model_order
        ]
    )

    components = _build_components(union)
    output: Set[Span] = set()

    for component in components:
        candidates = []

        for span in component:
            supporters = _get_supporters(
                span,
                model_spans,
                model_order,
            )

            families = {
                MODEL_FAMILY[model_name]
                for model_name in supporters
            }

            family_count = len(families)
            model_vote_count = len(supporters)

            if family_count < 2:
                continue

            candidates.append(
                (
                    span,
                    family_count,
                    model_vote_count,
                )
            )

        if not candidates:
            continue

        selected_span, _, _ = max(
            candidates,
            key=lambda item: (
                item[1],
                item[2],
                -_span_length(item[0]),
                -item[0][1],
                -item[0][2],
            ),
        )

        output.add(selected_span)

    return output


def _apply_safe_xlmr_additions(
    primary_prediction: Set[Span],
    model_spans: Dict[str, Set[Span]],
) -> Set[Span]:
    """
    Final XLM-R policy:

    1. Preserve the complete four-expert baseline.
    2. Build the family-aware prediction with XLM-R.
    3. Consider only additions caused by that policy.
    4. Addition must itself be supported by XLM-R.
    5. Addition must not overlap an existing baseline span of the same type.
    """
    family_aware_prediction = _build_family_aware_with_xlmr(
        model_spans
    )

    additions = (
        family_aware_prediction
        - primary_prediction
    )

    xlmr_supported_additions = {
        span
        for span in additions
        if span in model_spans["XLMR"]
    }

    safe_additions = {
        span
        for span in xlmr_supported_additions
        if not any(
            _overlaps(span, baseline_span)
            for baseline_span in primary_prediction
        )
    }

    return primary_prediction | safe_additions


# =============================================================================
# Main service
# =============================================================================

class NERInferenceService:
    """
    Raw-text wrapper around the final zero-KONOOZ NER ensemble.

    Training-only components are not used here.

    Inference-time components:
      - AraBERTv2 nested token expert
      - CAMeLBERT-Mix nested token expert
      - MARBERTv2 nested token expert
      - AraBERTv2 LIPN/span expert
      - XLM-R-base complementary expert

    The original research repository is treated as read-only.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._validate_paths()

        self._arabert_script = _load_module(
            NER_SCRIPTS
            / "infer_zero_konooz_arabert_clean_updated_v2_v1.py",
            "final_zero_konooz_arabert_inference",
        )

        self._span_script = _load_module(
            NER_SCRIPTS
            / "infer_span_zero_konooz_updated_v2_v1.py",
            "final_zero_konooz_span_inference",
        )

        self._load_primary_token_experts()
        self._load_span_expert()
        self._load_xlmr_expert()

    def _validate_paths(self) -> None:
        required = [
            ARABERT_MODEL_DIR,
            CAMELBERT_MODEL_DIR,
            MARBERTV2_MODEL_DIR,
            SPAN_CHECKPOINT,
            XLMR_MODEL_DIR,
            XLMR_CHECKPOINT,
            XLMR_TAG_VOCAB,
        ]

        missing = [
            path
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required final NER artifacts:\n"
                + "\n".join(str(path) for path in missing)
            )

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _load_primary_token_experts(self) -> None:
        from arabiner.utils.helpers import load_checkpoint

        self.token_experts = {}

        model_dirs = {
            "AraBERT": ARABERT_MODEL_DIR,
            "CAMeLBERT": CAMELBERT_MODEL_DIR,
            "MARBERTv2": MARBERTV2_MODEL_DIR,
        }

        for model_name, model_dir in model_dirs.items():
            tagger, tag_vocab, train_config = load_checkpoint(
                str(model_dir)
            )

            head_types = (
                self._arabert_script.identify_head_types(
                    tag_vocab
                )
            )

            if set(head_types) != set(ENTITY_ORDER):
                raise ValueError(
                    f"{model_name}: unexpected entity heads: "
                    f"{head_types}"
                )

            self.token_experts[model_name] = {
                "tagger": tagger,
                "tag_vocab": tag_vocab,
                "train_config": train_config,
                "head_types": head_types,
            }

    def _load_span_expert(self) -> None:
        from transformers import AutoTokenizer

        checkpoint = torch.load(
            SPAN_CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )

        labels = checkpoint["labels"]

        if labels != ENTITY_ORDER:
            raise ValueError(
                "Span checkpoint entity-type order does not "
                "match the final 21-type inventory."
            )

        args = checkpoint["args"]
        stored_model_name = args["model_name"]
        candidate_local = NER_ROOT / stored_model_name

        if candidate_local.exists():
            model_name = str(candidate_local)
        else:
            model_name = stored_model_name

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        model = self._span_script.MultiLabelLIPN(
            model_name=model_name,
            num_labels=len(ENTITY_ORDER),
            max_width=args["max_width"],
        ).to(self.device)

        model.load_state_dict(
            checkpoint["model_state"]
        )
        model.eval()

        collator = self._span_script.SpanCollator(
            tokenizer=tokenizer,
            max_width=args["max_width"],
            num_labels=len(ENTITY_ORDER),
        )

        self.span_model = model
        self.span_collator = collator

    def _load_xlmr_expert(self) -> None:
        from arabiner.bin import train as train_module
        from arabiner.utils.data import (
            get_dataloaders,
            text2segments,
        )

        self._xlmr_train_module = train_module
        self._xlmr_get_dataloaders = get_dataloaders
        self._xlmr_text2segments = text2segments

        with XLMR_TAG_VOCAB.open("rb") as handle:
            tag_vocab = pickle.load(handle)

        num_labels = [
            len(head_vocab)
            for head_vocab in tag_vocab[1:]
        ]

        if num_labels != [3] * 21:
            raise ValueError(
                "XLM-R tag vocabulary does not contain "
                "21 BIO heads with three classes each."
            )

        head_types = (
            self._arabert_script.identify_head_types(
                tag_vocab
            )
        )

        if set(head_types) != set(ENTITY_ORDER):
            raise ValueError(
                "XLM-R entity heads do not match the final "
                "21-type inventory."
            )

        network_config = {
            "fn": "arabiner.nn.AutoNestedTagger",
            "kwargs": {
                "dropout": 0.1,
                "bert_model": str(XLMR_MODEL_DIR),
                "num_labels": num_labels,
            },
        }

        model = train_module.load_object(
            network_config["fn"],
            network_config["kwargs"],
        )

        model = torch.nn.DataParallel(
            model,
            device_ids=[0],
        )

        if torch.cuda.is_available():
            model = model.cuda()

        checkpoint = torch.load(
            XLMR_CHECKPOINT,
            map_location=(
                None
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
            weights_only=False,
        )

        if "model" not in checkpoint:
            raise KeyError(
                "XLM-R checkpoint does not contain "
                "the 'model' key."
            )

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
        model.eval()

        self.xlmr_model = model
        self.xlmr_tag_vocab = tag_vocab
        self.xlmr_head_types = head_types

        self.xlmr_data_config = {
            "fn": "arabiner.data.auto_datasets.AutoNestedTagsDataset",
            "kwargs": {
                "max_seq_len": 512,
                "bert_model": str(XLMR_MODEL_DIR),
            },
        }

        # Build one harmless loader so BertNestedTrainer can be
        # initialized exactly as in the original evaluation script.
        dummy_loader = self._make_xlmr_loader(["اختبار"])

        optimizer = torch.optim.AdamW(
            self.xlmr_model.parameters(),
            lr=5e-6,
        )

        scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=1.0,
            )
        )

        loss = torch.nn.CrossEntropyLoss()

        self.xlmr_trainer = train_module.load_object(
            "arabiner.trainers.BertNestedTrainer",
            {
                "model": self.xlmr_model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "loss": loss,
                "train_dataloader": dummy_loader,
                "val_dataloader": dummy_loader,
                "test_dataloader": dummy_loader,
                "max_epochs": 1,
                "patience": 1,
                "clip": 1.0,
                "log_interval": 50,
                "summary_writer": None,
                "output_path": None,
            },
        )

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    @staticmethod
    def _tokenize_with_offsets(
        text: str,
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Match the whitespace-token input contract used by the existing
        inference scripts while retaining offsets into the original paragraph.
        """
        matches = list(re.finditer(r"\S+", text))

        tokens = [
            match.group(0)
            for match in matches
        ]

        offsets = [
            (match.start(), match.end())
            for match in matches
        ]

        return tokens, offsets

    # -------------------------------------------------------------------------
    # Primary token experts
    # -------------------------------------------------------------------------

    def _predict_primary_token_expert(
        self,
        model_name: str,
        tokens: Sequence[str],
    ) -> Set[Span]:
        expert = self.token_experts[model_name]

        lines = self._arabert_script.predict_sentence(
            list(tokens),
            expert["tagger"],
            expert["tag_vocab"],
            expert["train_config"],
            expert["head_types"],
        )

        tags = _prediction_lines_to_tags(lines)

        return _tags_to_spans(tags)

    # -------------------------------------------------------------------------
    # Span expert
    # -------------------------------------------------------------------------

    def _predict_span_expert(
        self,
        tokens: Sequence[str],
    ) -> Set[Span]:
        tags, _ = self._span_script.predict_sentence(
            list(tokens),
            self.span_model,
            self.span_collator,
            self.device,
        )

        return _tags_to_spans(tags)

    # -------------------------------------------------------------------------
    # XLM-R
    # -------------------------------------------------------------------------

    def _make_xlmr_loader(
        self,
        tokens: Sequence[str],
    ):
        text = " ".join(tokens)

        dataset, token_vocab = (
            self._xlmr_text2segments(text)
        )

        Vocab = namedtuple(
            "Vocab",
            ["tags", "tokens"],
        )

        vocab = Vocab(
            tokens=token_vocab,
            tags=self.xlmr_tag_vocab,
        )

        return self._xlmr_get_dataloaders(
            (dataset,),
            vocab,
            self.xlmr_data_config,
            batch_size=1,
            num_workers=0,
            shuffle=(False,),
        )[0]

    def _predict_xlmr(
        self,
        tokens: Sequence[str],
    ) -> Set[Span]:
        loader = self._make_xlmr_loader(tokens)

        with torch.inference_mode():
            _, segments, _, _ = (
                self.xlmr_trainer.eval(loader)
            )

        if len(segments) != 1:
            raise ValueError(
                "XLM-R expected exactly one predicted sentence."
            )

        predicted_tokens = segments[0]

        if len(predicted_tokens) != len(tokens):
            raise ValueError(
                "XLM-R changed token count: "
                f"input={len(tokens)}, "
                f"prediction={len(predicted_tokens)}"
            )

        tags: List[List[str]] = []

        for original_token, predicted_token in zip(
            tokens,
            predicted_tokens,
        ):
            if predicted_token.text != original_token:
                raise ValueError(
                    "XLM-R changed token: "
                    f"{original_token!r} -> "
                    f"{predicted_token.text!r}"
                )

            predictions_by_type = {
                entity_type:
                    predicted_token.pred_tag[index]["tag"]
                for index, entity_type
                in enumerate(self.xlmr_head_types)
            }

            ordered_tags = [
                predictions_by_type[entity_type]
                for entity_type in ENTITY_ORDER
            ]

            tags.append(ordered_tags)

        return _tags_to_spans(tags)

    # -------------------------------------------------------------------------
    # Public inference
    # -------------------------------------------------------------------------

    def predict(
        self,
        text: str,
    ) -> List[NEREntity]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return []

        tokens, offsets = self._tokenize_with_offsets(
            text
        )

        if not tokens:
            return []

        model_spans: Dict[str, Set[Span]] = {
            "AraBERT":
                self._predict_primary_token_expert(
                    "AraBERT",
                    tokens,
                ),
            "CAMeLBERT":
                self._predict_primary_token_expert(
                    "CAMeLBERT",
                    tokens,
                ),
            "MARBERTv2":
                self._predict_primary_token_expert(
                    "MARBERTv2",
                    tokens,
                ),
            "LIPN":
                self._predict_span_expert(tokens),
            "XLMR":
                self._predict_xlmr(tokens),
        }

        baseline = _build_primary_consensus(
            model_spans
        )

        final_spans = _apply_safe_xlmr_additions(
            baseline,
            model_spans,
        )

        all_model_order = MODEL_ORDER + ["XLMR"]

        entities: List[NEREntity] = []

        for entity_type, start, end in sorted(
            final_spans,
            key=lambda span: (
                span[1],
                span[2],
                ENTITY_ORDER.index(span[0]),
            ),
        ):
            if (
                start < 0
                or end <= start
                or end > len(tokens)
            ):
                raise ValueError(
                    f"Invalid final span: "
                    f"{(entity_type, start, end)}"
                )

            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            supporters = _get_supporters(
                (
                    entity_type,
                    start,
                    end,
                ),
                model_spans,
                all_model_order,
            )

            entities.append(
                NEREntity(
                    text=text[char_start:char_end],
                    entity_type=entity_type,
                    token_start=start,
                    token_end=end,
                    char_start=char_start,
                    char_end=char_end,
                    supporters=supporters,
                )
            )

        return entities

    def predict_dicts(
        self,
        text: str,
    ) -> List[dict]:
        return [
            {
                "text": entity.text,
                "type": entity.entity_type,
                "token_start": entity.token_start,
                "token_end": entity.token_end,
                "char_start": entity.char_start,
                "char_end": entity.char_end,
                "supporters": list(entity.supporters),
            }
            for entity in self.predict(text)
        ]


__all__ = [
    "NEREntity",
    "NERInferenceService",
]
