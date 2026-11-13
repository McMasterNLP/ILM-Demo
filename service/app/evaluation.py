from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import embed_query
from app.questions import GenerationUnavailable, _ask_raw
from app.repositories import retrieval as retrieval_repo

@dataclass(slots=True)
class Judgement:
    status: str
    is_correct: bool
    feedback: str
    evidence: list[str]

JUDGE_PROMPT = """You are grading a child's answer to a religious story question.

CRITICAL EVIDENCE RULE:
Use ONLY the APPROVED EVIDENCE below. The evidence is the complete authority for
this grading decision. Do not use your own knowledge of Islam, the Quran, hadith,
tafsir, history, prophets, names, events, places, or any other outside knowledge.

Do not add, infer, complete, correct, or introduce a religious fact unless it is
supported by the APPROVED EVIDENCE.

Compare the child's answer with the question, the approved evidence, and, when
provided, the reference answer.

Return exactly one status:

"correct"
- The meaning of the child's answer is supported by the approved evidence and
  answers the question.

"incorrect"
- The approved evidence establishes the answer, but the child's answer
  contradicts it or materially misses the required meaning.

"not_enough_evidence"
- The approved evidence does not establish enough information to decide safely.
- Use this instead of guessing.
- Never mark an answer correct merely because the evidence does not contradict it.

Be generous about wording, spelling and grammar. Different wording is acceptable
when the same meaning is supported by the evidence.

The reference answer is guidance for the expected answer, but it is NOT permission
to introduce facts absent from the approved evidence.

Write one short feedback sentence in {language}.
For "not_enough_evidence", say that the answer cannot be verified from the
approved story evidence. Do not invent the missing information.

QUESTION: {question}
CHILD'S ANSWER: {answer}
{reference}

Return only JSON:
{{"status": "correct|incorrect|not_enough_evidence", "feedback": "..."}}

APPROVED EVIDENCE:
{passages}
"""

LANGUAGES = {"en": "English", "ar": "Arabic"}


def _ask_openrouter(message: str) -> str:
    """
    Semantic grading only.

    Question generation remains on the existing Gemini wrapper.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise GenerationUnavailable("OPENROUTER_API_KEY is not configured")

    model = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
    if not model:
        model = "openrouter/free"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ],
        "temperature": 0,
    }

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise GenerationUnavailable(
            f"OpenRouter grading unavailable: HTTP {exc.code}: {body}"
        ) from exc

    except Exception as exc:
        raise GenerationUnavailable(
            f"OpenRouter grading unavailable: {exc}"
        ) from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GenerationUnavailable(
            "OpenRouter returned an unexpected response"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise GenerationUnavailable(
            "OpenRouter returned an empty grading response"
        )

    content = content.strip()

    # Free models sometimes wrap JSON in Markdown fences.
    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    # Preserve judge_answer()'s existing json.loads(raw) behavior.
    # If the model adds harmless surrounding prose, isolate the object.
    try:
        json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise GenerationUnavailable(
                "OpenRouter grader did not return valid JSON"
            )

        content = content[start:end + 1]

        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise GenerationUnavailable(
                "OpenRouter grader did not return valid JSON"
            ) from exc

    return content

def judge_answer(
    session: Session,
    *,
    story_id: str,
    question: str,
    learner_answer: str,
    reference_answer: str = "",
    lang: str = "en",
    age_range: str = "7-10",
    top_k: int = 3,
) -> Judgement:
    answer = learner_answer.strip()
    if not answer:
        return Judgement(status="incorrect", is_correct=False, feedback="", evidence=[])

    vector = embed_query(f"{question}\n{answer}")
    hits = retrieval_repo.similar_chunks(
        session,
        vector,
        model=settings.embedding_model,
        story_id=story_id,
        lang=lang,
        limit=top_k,
    )
    if not hits:
        raise GenerationUnavailable("No indexed passages for this story")

    passages = "\n\n".join(f"- {hit.chunk.text}" for hit in hits)
    reference = (
        f"A GOOD ANSWER WOULD BE: {reference_answer}\n" if reference_answer else ""
    )

    raw = _ask_openrouter(
        JUDGE_PROMPT.format(
            age=age_range,
            language=LANGUAGES.get(lang, "English"),
            question=question,
            answer=answer,
            reference=reference,
            passages=passages,
        )
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenerationUnavailable(f"Judge did not return JSON: {raw[:200]}") from exc

    status = str(data.get("status", "")).strip().lower()
    allowed = {"correct", "incorrect", "not_enough_evidence"}

    if status not in allowed:
        raise GenerationUnavailable(
            f"Judge returned invalid status: {status or '<missing>'}"
        )

    return Judgement(
        status=status,
        is_correct=status == "correct",
        feedback=str(data.get("feedback", "")).strip(),
        evidence=[hit.chunk.id for hit in hits],
    )
