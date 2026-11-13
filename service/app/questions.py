from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from app.config import settings
from app.embeddings import normalize

class GenerationUnavailable(RuntimeError):
    pass

_TRANSIENT = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")
_EXHAUSTED = ("PerDay", "per day", "GenerateRequestsPerDay")
_ATTEMPTS = 3
_BACKOFF = 2.0

_client = None

def _get_client():
    global _client
    if not settings.gemini_api_key:
        raise GenerationUnavailable(
            "No GEMINI_API_KEY configured — question generation is disabled."
        )
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client

@dataclass(slots=True)
class GeneratedOpenQuestion:
    prompt: str
    reference_answer: str
    explanation: str

@dataclass(slots=True)
class GeneratedQuestion:
    prompt: str
    options: list[str]
    correct_index: int
    explanation: str

PROMPT = """You write reading-comprehension questions for children aged {age}.

Use ONLY the SOURCE below. Do not add names, events, places or details that are
not in it. If the SOURCE does not say something, do not ask about it.

Write ONE multiple-choice question that a child can answer from the SOURCE alone.
Give exactly 4 options. Exactly one is correct; the other three must be clearly
wrong but plausible, and must also stay within the world of the SOURCE.

Ask about the story itself, never about the writing. Do not use phrases like
"according to the text", "in the passage", "the source says" or "the story
states" — not in the question, the options, or the explanation. A child is being
asked what happened, not what a document contains.

Write the explanation as one plain sentence saying what happened, not as a
report about the text. Say "Adam sneezed and said Alhamdulillah, his first
words." — not "The text states that Adam said Alhamdulillah."

Write the question, the options and the explanation in {language}.
{avoid}
Return only JSON:
{{"prompt": "...", "options": ["...","...","...","..."], "correct_index": 0, "explanation": "..."}}

SOURCE:
{source}
"""

FOLLOWUP_PROMPT = """A child aged {age} answered this question wrongly:

QUESTION: {prompt}
CORRECT ANSWER: {answer}

Ask the SAME fact again in a different way, so they can show they now
understand it. Do not reuse the original wording. Do not make it easier by
hinting at the answer.

Use ONLY the SOURCE below. Give exactly 3 options, exactly one correct.

Ask about the story itself, never about the writing. Do not use phrases like
"according to the text" or "the story states".

Write everything in {language}.

Return only JSON:
{{"prompt": "...", "options": ["...","...","..."], "correct_index": 0, "explanation": "..."}}

SOURCE:
{source}
"""

OPEN_PROMPT = """You write reading-comprehension questions for children aged {age}.

Use ONLY the SOURCE below. Do not add names, events, places or details that are
not in it. If the SOURCE does not say something, do not ask about it.

Write ONE question that a child answers in their own words, in a sentence or
two. It must be answerable from the SOURCE alone.

Ask about the story itself, never about the writing. Do not use phrases like
"according to the text" or "the story states".

Also give the answer you would expect, in one short sentence a child might write.

Write everything in {language}.
{avoid}
Return only JSON:
{{"prompt": "...", "reference_answer": "...", "explanation": "..."}}

SOURCE:
{source}
"""

AVOID_BLOCK = """
These questions have already been asked about this same passage. Ask about
something else in it — a different detail, person, action or consequence. Do
not rephrase any of them:
{existing}
"""

LANGUAGES = {"en": "English", "ar": "Arabic"}

def _parse(text: str, expected_options: int = 4) -> GeneratedQuestion:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationUnavailable(f"Model did not return JSON: {text[:200]}") from exc

    if not isinstance(data, dict):
        raise GenerationUnavailable("Expected a question object")

    prompt = str(data.get("prompt", "")).strip()
    options = [str(o).strip() for o in data.get("options", []) if str(o).strip()]
    explanation = str(data.get("explanation", "")).strip()

    try:
        correct = int(data.get("correct_index", -1))
    except (TypeError, ValueError) as exc:
        raise GenerationUnavailable("correct_index is not a number") from exc

    if not prompt:
        raise GenerationUnavailable("Model returned no question")
    if len(options) != expected_options:
        raise GenerationUnavailable(
            f"Expected {expected_options} options, got {len(options)}"
        )
    if len(set(options)) != expected_options:
        raise GenerationUnavailable("Options are not distinct")
    if not 0 <= correct < expected_options:
        raise GenerationUnavailable(f"correct_index {correct} out of range")

    return GeneratedQuestion(
        prompt=prompt, options=options, correct_index=correct, explanation=explanation
    )

def _ask_raw(message: str) -> str:
    client = _get_client()

    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=message,
                config={"response_mime_type": "application/json", "temperature": 0.3},
            )
        except Exception as exc:
            last = exc
            detail = str(exc)
            if any(marker in detail for marker in _EXHAUSTED):
                raise GenerationUnavailable(
                    f"{settings.gemini_model}: daily quota exhausted. Free-tier keys "
                    "allow a limited number of requests per day — wait for the reset "
                    "or use a billed key."
                ) from exc
            if not any(marker in detail for marker in _TRANSIENT):
                raise GenerationUnavailable(f"{settings.gemini_model}: {detail}") from exc
            if attempt < _ATTEMPTS - 1:
                time.sleep(_BACKOFF * (2**attempt))
            continue

        text = (response.text or "").strip()
        if not text:
            raise GenerationUnavailable("Model returned an empty response")
        return text

    raise GenerationUnavailable(
        f"{settings.gemini_model} is unavailable after {_ATTEMPTS} attempts: {last}"
    )

def _ask(message: str, expected_options: int) -> GeneratedQuestion:
    return _parse(_ask_raw(message), expected_options)

def _avoid(existing: list[str] | None) -> str:
    if not existing:
        return ""
    joined = chr(10).join(f"- {q}" for q in existing)
    return AVOID_BLOCK.format(existing=joined)

def generate_question(
    chunk_text: str,
    lang: str,
    age_range: str = "7-10",
    existing: list[str] | None = None,
) -> GeneratedQuestion:
    return _ask(
        PROMPT.format(
            age=age_range,
            language=LANGUAGES.get(lang, "English"),
            source=chunk_text,
            avoid=_avoid(existing),
        ),
        expected_options=4,
    )

def generate_followup(
    chunk_text: str,
    original: GeneratedQuestion,
    lang: str,
    age_range: str = "7-10",
) -> GeneratedQuestion:
    return _ask(
        FOLLOWUP_PROMPT.format(
            age=age_range,
            language=LANGUAGES.get(lang, "English"),
            prompt=original.prompt,
            answer=original.options[original.correct_index],
            source=chunk_text,
        ),
        expected_options=3,
    )

_WORD = re.compile(r"\w+", re.UNICODE)

def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(normalize(text).lower()) if len(t) > 2}

def looks_grounded(question: GeneratedQuestion, chunk_text: str) -> bool:
    answer = _tokens(question.options[question.correct_index])
    if not answer:
        needle = normalize(question.options[question.correct_index]).lower()
        return bool(needle) and needle in normalize(chunk_text).lower()

    source = _tokens(chunk_text)
    return len(answer & source) / len(answer) >= 0.5

def generate_open_question(
    chunk_text: str,
    lang: str,
    age_range: str = "7-10",
    existing: list[str] | None = None,
) -> GeneratedOpenQuestion:
    raw = _ask_raw(
        OPEN_PROMPT.format(
            age=age_range,
            language=LANGUAGES.get(lang, "English"),
            source=chunk_text,
            avoid=_avoid(existing),
        )
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenerationUnavailable(f"Model did not return JSON: {raw[:200]}") from exc

    prompt = str(data.get("prompt", "")).strip()
    reference = str(data.get("reference_answer", "")).strip()
    if not prompt:
        raise GenerationUnavailable("Model returned no question")
    if not reference:
        raise GenerationUnavailable("Model returned no reference answer")

    return GeneratedOpenQuestion(
        prompt=prompt,
        reference_answer=reference,
        explanation=str(data.get("explanation", "")).strip(),
    )

_DUPLICATE_AT = 0.6

def is_duplicate(prompt: str, existing: list[str]) -> bool:
    new = _tokens(prompt)
    if not new:
        return False
    for other in existing:
        old = _tokens(other)
        if not old:
            continue
        overlap = len(new & old) / len(new | old)
        if overlap >= _DUPLICATE_AT:
            return True
    return False
