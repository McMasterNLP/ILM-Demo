from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import error, request

from app.ml.kg_records import (
    KGEntityMentionRecord,
    KGEntityRecord,
    KGPreparedRecords,
    KGRelationRecord,
    KGStoryEntityRecord,
    KGStoryRelationRecord,
)


DEFAULT_ML_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 300


class KGBridgeError(RuntimeError):
    """Raised when the external KG/ML service fails or returns invalid data."""


@dataclass(frozen=True)
class KGBridgeResult:
    story_id: str
    story_key: str
    paragraph_id: str
    lang: str
    records: KGPreparedRecords
    diagnostics: dict[str, int]


def _deserialize_records(payload: dict) -> KGPreparedRecords:
    try:
        return KGPreparedRecords(
            entities=tuple(
                KGEntityRecord(**item)
                for item in payload.get("entities", [])
            ),
            mentions=tuple(
                KGEntityMentionRecord(**item)
                for item in payload.get("mentions", [])
            ),
            story_entities=tuple(
                KGStoryEntityRecord(**item)
                for item in payload.get("story_entities", [])
            ),
            relations=tuple(
                KGRelationRecord(**item)
                for item in payload.get("relations", [])
            ),
            story_relations=tuple(
                KGStoryRelationRecord(**item)
                for item in payload.get("story_relations", [])
            ),
        )
    except (TypeError, KeyError) as exc:
        raise KGBridgeError(
            f"Invalid KG records returned by ML service: {exc}"
        ) from exc


def run_kg_extraction(
    *,
    story_id: str,
    story_key: str,
    paragraph_id: str,
    lang: str,
    text: str,
) -> KGBridgeResult:
    """
    Call the external KG/ML HTTP service and convert its JSON response
    into the lightweight KG record objects used by the application.

    This module intentionally does not import torch, transformers,
    NER, or RE.
    """

    if not story_id:
        raise ValueError("story_id is required")
    if not story_key:
        raise ValueError("story_key is required")
    if not paragraph_id:
        raise ValueError("paragraph_id is required")
    if not lang:
        raise ValueError("lang is required")
    if not text or not text.strip():
        raise ValueError("text is required")

    ml_url = os.getenv(
        "KG_ML_URL",
        DEFAULT_ML_URL,
    ).rstrip("/")

    try:
        timeout_seconds = int(
            os.getenv(
                "KG_ML_TIMEOUT_SECONDS",
                str(DEFAULT_TIMEOUT_SECONDS),
            )
        )
    except ValueError as exc:
        raise KGBridgeError(
            "KG_ML_TIMEOUT_SECONDS must be an integer"
        ) from exc

    request_payload = {
        "story_id": story_id,
        "story_key": story_key,
        "paragraph_id": paragraph_id,
        "lang": lang,
        "text": text,
    }

    body = json.dumps(
        request_payload,
        ensure_ascii=False,
    ).encode("utf-8")

    http_request = request.Request(
        url=f"{ml_url}/extract",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
    )

    try:
        with request.urlopen(
            http_request,
            timeout=timeout_seconds,
        ) as response:
            raw_response = response.read()

    except error.HTTPError as exc:
        try:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            detail = ""

        raise KGBridgeError(
            f"KG ML service returned HTTP {exc.code}"
            + (f": {detail}" if detail else "")
        ) from exc

    except error.URLError as exc:
        raise KGBridgeError(
            f"Unable to reach KG ML service at {ml_url}: "
            f"{exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise KGBridgeError(
            f"KG extraction exceeded {timeout_seconds} seconds"
        ) from exc

    try:
        response = json.loads(
            raw_response.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise KGBridgeError(
            "KG ML service returned invalid JSON"
        ) from exc

    if not isinstance(response, dict):
        raise KGBridgeError(
            "KG ML response must be a JSON object"
        )

    if "error" in response:
        raise KGBridgeError(
            "KG ML service returned an error: "
            f"{response.get('message', response['error'])}"
        )

    required = {
        "story_id",
        "story_key",
        "paragraph_id",
        "lang",
        "records",
        "diagnostics",
    }

    missing = required.difference(response)
    if missing:
        raise KGBridgeError(
            "KG ML response is missing field(s): "
            + ", ".join(sorted(missing))
        )

    expected_identity = (
        story_id,
        story_key,
        paragraph_id,
        lang,
    )

    returned_identity = (
        response["story_id"],
        response["story_key"],
        response["paragraph_id"],
        response["lang"],
    )

    if returned_identity != expected_identity:
        raise KGBridgeError(
            "KG ML response identity does not match request"
        )

    if not isinstance(response["records"], dict):
        raise KGBridgeError(
            "KG ML response 'records' must be an object"
        )

    diagnostics = response["diagnostics"]

    if not isinstance(diagnostics, dict):
        raise KGBridgeError(
            "KG ML response 'diagnostics' must be an object"
        )

    return KGBridgeResult(
        story_id=response["story_id"],
        story_key=response["story_key"],
        paragraph_id=response["paragraph_id"],
        lang=response["lang"],
        records=_deserialize_records(
            response["records"]
        ),
        diagnostics=diagnostics,
    )
