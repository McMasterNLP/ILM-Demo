from pathlib import Path

path = Path(r"service/app/evaluation.py")
text = path.read_text(encoding="utf-8")

if "def _ask_openrouter(" in text:
    print("OpenRouter grader already patched")
    raise SystemExit(0)

text = text.replace(
    "from dataclasses import dataclass\n",
    "from dataclasses import dataclass\nimport json\nimport os\nimport urllib.error\nimport urllib.request\n",
    1,
)

helper = '''

def _ask_openrouter(message: str) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise GenerationUnavailable("OPENROUTER_API_KEY is not configured")

    model = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
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
        content = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise GenerationUnavailable(
            "OpenRouter returned an unexpected response"
        ) from exc

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\\n".join(lines).strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise GenerationUnavailable(
                "OpenRouter grader did not return valid JSON"
            )
        result = json.loads(content[start:end + 1])

    return result
'''

marker = "\ndef judge_answer("
if marker not in text:
    raise RuntimeError("judge_answer marker not found")

text = text.replace(marker, helper + marker, 1)

if "_ask_raw(prompt)" not in text:
    raise RuntimeError("_ask_raw(prompt) not found")

text = text.replace("_ask_raw(prompt)", "_ask_openrouter(prompt)", 1)

path.write_text(text, encoding="utf-8")
print("OPENROUTER GRADER PATCH: OK")
