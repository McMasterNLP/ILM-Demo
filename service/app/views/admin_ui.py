from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import settings

router = APIRouter()

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ilm - Authoring</title>
<style>
  :root {
    --brand: #7A003C; --deep: #63002F; --tint: #F6EBF1;
    --canvas: #F4F2F0; --hair: #E7E5E4;
    --ink: #1C1917; --soft: #57534E; --muted: #A8A29E;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--canvas); color: var(--ink);
    font: 14px/1.5 Inter, ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  }
  header {
    background: #fff; border-bottom: 1px solid var(--hair);
    padding: 16px 24px; display: flex; align-items: center; justify-content: space-between;
  }
  header strong { font-size: 16px; }
  header a { color: var(--soft); text-decoration: none; font-size: 13px; margin-left: 16px; }
  header a:hover { color: var(--brand); }
  main { max-width: 900px; margin: 0 auto; padding: 28px 24px 80px; }
  section {
    background: #fff; border: 1px solid var(--hair); border-radius: 12px;
    padding: 20px; margin-bottom: 18px;
  }
  section[hidden] { display: none; }
  h2 { font-size: 15px; margin: 0 0 4px; }
  .eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: var(--brand); margin: 0 0 10px;
  }
  p.hint { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
  label { display: block; font-size: 12px; color: var(--soft); margin: 12px 0 4px; }
  input, select, textarea {
    width: 100%; padding: 8px 10px; border: 1px solid var(--hair);
    border-radius: 8px; font: inherit; background: #fff;
  }
  textarea {
    min-height: 260px; resize: vertical; line-height: 1.8;
    font-family: inherit;
  }
  .toggle { display: inline-flex; border: 1px solid var(--hair); border-radius: 8px; overflow: hidden; }
  .toggle button {
    border: 0; border-radius: 0; color: var(--soft); background: #fff; padding: 8px 16px;
  }
  .toggle button.on { background: var(--tint); color: var(--brand); font-weight: 500; }
  .counter { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; }
  .row > * { flex: 1; min-width: 150px; }
  button {
    font: inherit; cursor: pointer; border-radius: 8px; padding: 9px 16px;
    border: 1px solid var(--brand); background: #fff; color: var(--brand);
    transition: background .15s;
  }
  button:hover:not(:disabled) { background: var(--tint); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button.primary { background: var(--brand); color: #fff; }
  button.primary:hover:not(:disabled) { background: var(--deep); }
  button.quiet { border-color: var(--hair); color: var(--soft); }
  .actions { margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .para {
    border-top: 1px solid var(--hair); padding: 14px 0;
    display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: start;
  }
  .para:first-of-type { border-top: 0; }
  .body { font-size: 13px; line-height: 1.75; white-space: pre-wrap; }
  .meta { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
  .tag {
    font-size: 11px; padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--hair); color: var(--soft); white-space: nowrap;
  }
  .tag.approved { background: var(--tint); border-color: var(--brand); color: var(--brand); }
  .q { border-top: 1px solid var(--hair); padding: 14px 0; }
  .q:first-of-type { border-top: 0; }
  .q h3 { font-size: 14px; margin: 6px 0 8px; font-weight: 500; }
  .q ol { margin: 0 0 10px; padding-inline-start: 20px; font-size: 13px; color: var(--soft); }
  .q li.right { color: var(--brand); font-weight: 500; }
  .warn {
    background: #FDF2F2; border: 1px solid #F0C9C9; color: #B4232C;
    border-radius: 8px; padding: 10px 12px; font-size: 12px; margin-bottom: 12px;
  }
  .status { font-size: 13px; color: var(--soft); margin-top: 12px; min-height: 20px; }
  .status.bad { color: #B4232C; }
  ul.stories { list-style: none; margin: 0; padding: 0; }
  ul.stories li {
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
    padding: 10px 0; border-top: 1px solid var(--hair); font-size: 13px;
  }
  ul.stories li:first-child { border-top: 0; }
  code { font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <strong>Ilm - Authoring</strong>
  <nav>
    <a href="/admin/">Curation tables</a>
    <a href="/admin/logout">Sign out</a>
  </nav>
</header>

<main>
  <section>
    <p class="eyebrow">Step 1 - Story</p>
    <h2>Add a written story</h2>
    <p class="hint">
      Blank lines separate paragraphs - one paragraph becomes one retrievable
      chunk, so keep them as you want them read.
    </p>

    <div class="toggle" id="modeToggle">
      <button type="button" data-mode="paste" class="on">Paste text</button>
      <button type="button" data-mode="file">Upload file</button>
    </div>

    <label for="title">Story title</label>
    <input id="title" placeholder="e.g. Prophet Yusuf (AS)" />

    <label for="summary">Summary</label>
    <input id="summary" placeholder="One line for the library card" />

    <div class="row">
      <div>
        <label for="lang">Language</label>
        <select id="lang"><option value="en">English</option><option value="ar">العربية</option></select>
      </div>
      <div>
        <label for="level">Level</label>
        <select id="level">
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </select>
      </div>
      <div>
        <label for="minutes">Minutes (0 = estimate)</label>
        <input id="minutes" type="number" value="0" min="0" />
      </div>
    </div>

    <label for="sourceTitle">Source it came from</label>
    <input id="sourceTitle" placeholder="e.g. Stories of the Prophets, Ibn Kathir" />

    <div id="pasteMode">
      <label for="text">Story text</label>
      <textarea id="text" placeholder="Paste the story here.&#10;&#10;Leave a blank line between paragraphs."></textarea>
      <p class="counter" id="counter">0 paragraphs</p>
    </div>

    <div id="fileMode" hidden>
      <label for="file">File</label>
      <input id="file" type="file" accept=".pdf,.txt,.md,.markdown" />
    </div>

    <div class="actions">
      <button class="primary" id="upload">Save story</button>
      <span class="status" id="uploadStatus"></span>
    </div>
  </section>

  <section>
    <p class="eyebrow">Library</p>
    <h2>Uploaded stories</h2>
    <ul class="stories" id="stories"></ul>
  </section>

  <section id="reviewSection" hidden>
    <p class="eyebrow">Step 2 - Check</p>
    <h2 id="reviewTitle">Paragraphs</h2>
    <p class="hint">
      How it split. Approved paragraphs are indexed for retrieval automatically.
      Send one back if it came out wrong - it then reads as pending and drops
      out of search too.
    </p>
    <div id="review"></div>
    <div class="actions">
      <button class="primary" id="publish">Publish story</button>
      <button class="quiet" id="unpublish">Unpublish</button>
      <span class="status" id="publishStatus"></span>
    </div>
  </section>

  <section id="quizSection" hidden>
    <p class="eyebrow">Step 3 - Quiz</p>
    <h2>Generate questions</h2>
    <p class="hint">
      Each question is written from one indexed paragraph and nothing else, then
      shown beside that paragraph so you can check it before children see it.
      Written answers are graded against passages retrieved at the moment the
      child answers.
      <br /><strong>Generate more</strong> adds to what is here, covering the
      least-asked-about paragraphs first and avoiding repeats.
      <strong>Cover every paragraph</strong> fills the gaps instead, ignoring
      the count. <strong>Start over</strong> discards questions still awaiting
      review. Approved questions are never touched by any of them.
      <br />A quiz shows only a handful of these - the bank covers the story,
      each sitting samples from it.
    </p>

    <div class="row">
      <div>
        <label for="qCount">How many</label>
        <input id="qCount" type="number" value="6" min="1" max="20" />
      </div>
      <div>
        <label for="qAge">Age range</label>
        <input id="qAge" value="7-10" />
      </div>
      <div>
        <label for="qMode">Answer style</label>
        <select id="qMode">
          <option value="choice">Multiple choice</option>
          <option value="open">Write an answer</option>
        </select>
      </div>
    </div>

    <div class="actions">
      <button class="primary" id="generate">Generate more</button>
      <button id="coverAll">Cover every paragraph</button>
      <button class="quiet" id="regenerate">Start over</button>
      <span class="status" id="quizStatus"></span>
    </div>
    <p class="counter" id="coverage"></p>

    <div id="warnings"></div>
    <div id="questions"></div>

    <div style="border-top:1px solid var(--hair);margin-top:18px;padding-top:16px">
      <p class="eyebrow">Quiz shape</p>
      <p class="hint">
        What one sitting shows. The bank above can be much larger - a child sees
        this many, drawn across different paragraphs. Leave the type counts at 0
        to let any approved question fill the quiz.
      </p>
      <div class="row">
        <div>
          <label for="shapeSize">Questions per quiz</label>
          <input id="shapeSize" type="number" min="1" max="50" value="6" />
        </div>
        <div>
          <label for="shapeRel">Relationship</label>
          <input id="shapeRel" type="number" min="0" max="50" value="0" />
        </div>
        <div>
          <label for="shapeIdent">Identification</label>
          <input id="shapeIdent" type="number" min="0" max="50" value="0" />
        </div>
        <div>
          <label for="shapeComp">Comprehension</label>
          <input id="shapeComp" type="number" min="0" max="50" value="0" />
        </div>
      </div>
      <div class="actions">
        <button id="saveShape">Save quiz shape</button>
        <span class="status" id="shapeStatus"></span>
      </div>
    </div>
  </section>
</main>

<script>
const API = "__API__/authoring";
const state = { storyId: null };
const $ = (id) => document.getElementById(id);

function say(el, message, bad) {
  el.textContent = message || "";
  el.classList.toggle("bad", Boolean(bad));
}

async function call(path, options = {}) {
  const response = await fetch(API + path, { credentials: "same-origin", ...options });
  if (response.status === 401) {
    window.location.href = "/admin/login";
    throw new Error("Signed out");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status})`);
  return body;
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

/* ---------------------------------------------------------------- input */

let mode = "paste";

for (const button of document.querySelectorAll("#modeToggle button")) {
  button.onclick = () => {
    mode = button.dataset.mode;
    for (const b of document.querySelectorAll("#modeToggle button")) {
      b.classList.toggle("on", b === button);
    }
    $("pasteMode").hidden = mode !== "paste";
    $("fileMode").hidden = mode !== "file";
    say($("uploadStatus"), "");
  };
}

// Same split the server does, so the count shown is the count stored.
function countParagraphs(text) {
  return text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean).length;
}

$("text").oninput = () => {
  const n = countParagraphs($("text").value);
  $("counter").textContent = `${n} paragraph${n === 1 ? "" : "s"}`;
};

$("upload").onclick = async () => {
  const file = $("file").files[0];
  const text = $("text").value;
  const title = $("title").value.trim();
  const sourceTitle = $("sourceTitle").value.trim();

  if (mode === "file" && !file) return say($("uploadStatus"), "Choose a file first", true);
  if (mode === "paste" && !text.trim()) return say($("uploadStatus"), "Paste the story first", true);
  if (!title) return say($("uploadStatus"), "Give the story a title", true);
  if (!sourceTitle) return say($("uploadStatus"), "Name the source it came from", true);

  const form = new FormData();
  if (mode === "file") form.append("file", file);
  else form.append("text", text);
  form.append("title", title);
  form.append("summary", $("summary").value.trim());
  form.append("lang", $("lang").value);
  form.append("level", $("level").value);
  form.append("minutes", $("minutes").value || "0");
  form.append("source_title", sourceTitle);

  $("upload").disabled = true;
  say($("uploadStatus"), mode === "file" ? "Reading the file…" : "Saving…");
  try {
    const review = await call("/stories", { method: "POST", body: form });
    say($("uploadStatus"), `Stored - ${review.total} paragraphs.`);
    state.storyId = review.story_id;
    renderReview(review);
    await loadStories();
  } catch (err) {
    say($("uploadStatus"), err.message, true);
  } finally {
    $("upload").disabled = false;
  }
};

/* ------------------------------------------------------------- library */

async function loadStories() {
  const documents = await call("/documents");
  const host = $("stories");
  host.innerHTML = "";
  const seen = new Set();

  for (const doc of documents) {
    if (!doc.story_id || seen.has(doc.story_id)) continue;
    seen.add(doc.story_id);

    const li = document.createElement("li");
    li.innerHTML =
      `<span>${escapeHtml(doc.story_id)} <code>${escapeHtml(doc.filename)} · ${doc.lang}</code></span>`;
    const button = document.createElement("button");
    button.className = "quiet";
    button.textContent = "Open";
    button.onclick = () => openStory(doc.story_id);
    li.appendChild(button);
    host.appendChild(li);
  }

  if (!host.children.length) {
    host.innerHTML = '<li><span class="tag">Nothing uploaded yet</span></li>';
  }
}

async function openStory(storyId) {
  state.storyId = storyId;
  renderReview(await call(`/stories/${storyId}/review`));
  loadQuestions().catch(() => {});
  $("reviewSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* -------------------------------------------------------------- review */

function renderReview(review) {
  state.storyId = review.story_id;
  $("reviewSection").hidden = false;
  $("quizSection").hidden = false;
  $("reviewTitle").textContent =
    `${review.title} - ${review.total} paragraphs, ${review.approved} approved, ` +
    `${review.embedded_count}/${review.chunk_count} indexed`;

  const host = $("review");
  host.innerHTML = "";

  for (const item of review.paragraphs) {
    const wrapper = document.createElement("div");
    wrapper.className = "para";

    const left = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML =
      `<span class="tag">¶ ${item.position + 1} · ${item.lang}</span>` +
      `<span class="tag ${item.status === "approved" ? "approved" : ""}">${item.status}</span>` +
      (item.approved_by ? `<code>by ${escapeHtml(item.approved_by)}</code>` : "");
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = item.body;
    left.append(meta, body);

    const button = document.createElement("button");
    button.className = "quiet";
    button.textContent = item.status === "approved" ? "Reject" : "Approve";
    button.onclick = () =>
      act(item, item.status === "approved" ? "reject" : "approve");

    wrapper.append(left, button);
    host.appendChild(wrapper);
  }

  $("shapeSize").value = review.quiz_size;
  $("shapeRel").value = review.quiz_relationship;
  $("shapeIdent").value = review.quiz_identification;
  $("shapeComp").value = review.quiz_comprehension;

  const allApproved = review.total > 0 && review.approved === review.total;
  const allIndexed = review.chunk_count > 0 && review.embedded_count === review.chunk_count;
  const ready = allApproved && allIndexed;

  $("publish").disabled = !ready || review.published;
  $("unpublish").disabled = !review.published;
  say(
    $("publishStatus"),
    review.published
      ? "Published - visible in the library."
      : ready
        ? "Approved and indexed. Ready to publish."
        : !allApproved
          ? `${review.total - review.approved} paragraph(s) sent back.`
          : `${review.chunk_count - review.embedded_count} paragraph(s) not indexed - check COHERE_API_KEY.`
  );
}

async function act(item, verb) {
  try {
    await call(`/paragraphs/${item.paragraph_id}/${item.lang}/${verb}`, { method: "POST" });
    renderReview(await call(`/stories/${state.storyId}/review`));
  } catch (err) {
    say($("publishStatus"), err.message, true);
  }
}

async function setPublished(published) {
  try {
    renderReview(await call(`/stories/${state.storyId}/publish`, json({ published })));
  } catch (err) {
    say($("publishStatus"), err.message, true);
  }
}

$("publish").onclick = () => setPublished(true);
$("unpublish").onclick = () => setPublished(false);

/* ---------------------------------------------------------------- quiz */

async function loadQuestions() {
  const lang = $("lang").value;
  renderQuestions(await call(`/stories/${state.storyId}/questions?lang=${lang}`));
}

async function generate(replace, coverAll) {
  $("generate").disabled = true;
  $("regenerate").disabled = true;
  $("coverAll").disabled = true;
  say($("quizStatus"), "Writing questions from the story…");
  try {
    const set = await call(
      `/stories/${state.storyId}/questions`,
      json({
        lang: $("lang").value,
        count: Number($("qCount").value) || 6,
        age_range: $("qAge").value.trim() || "7-10",
        answer_mode: $("qMode").value,
        replace,
        cover_all: Boolean(coverAll),
      })
    );
    say($("quizStatus"), `${set.total} question(s) - review them below.`);
    renderQuestions(set);
  } catch (err) {
    say($("quizStatus"), err.message, true);
  } finally {
    $("generate").disabled = false;
    $("regenerate").disabled = false;
    $("coverAll").disabled = false;
  }
}

$("saveShape").onclick = async () => {
  $("saveShape").disabled = true;
  try {
    const review = await call(`/stories/${state.storyId}/quiz-shape`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quiz_size: Number($("shapeSize").value) || 6,
        quiz_relationship: Number($("shapeRel").value) || 0,
        quiz_identification: Number($("shapeIdent").value) || 0,
        quiz_comprehension: Number($("shapeComp").value) || 0,
      }),
    });
    renderReview(review);
    say($("shapeStatus"), `A quiz will show ${review.quiz_size} question(s).`);
  } catch (err) {
    say($("shapeStatus"), err.message, true);
  } finally {
    $("saveShape").disabled = false;
  }
};

$("generate").onclick = () => generate(false, false);
$("coverAll").onclick = () => generate(false, true);
$("regenerate").onclick = () => {
  if (confirm("Discard the questions still awaiting review and start again?")) {
    generate(true, false);
  }
};

function renderQuestions(set) {
  $("generate").textContent = set.total > 0 ? "Generate more" : "Generate questions";

  const warn = $("warnings");
  warn.innerHTML = "";
  for (const message of set.warnings || []) {
    const div = document.createElement("div");
    div.className = "warn";
    div.textContent = message;
    warn.appendChild(div);
  }

  const host = $("questions");
  host.innerHTML = "";

  for (const q of set.questions) {
    const wrapper = document.createElement("div");
    wrapper.className = "q";

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML =
      `<span class="tag">${q.type}</span>` +
      `<span class="tag ${q.status === "approved" ? "approved" : ""}">${q.status}</span>` +
      (q.evidence_paragraph !== null ? `<code>¶ ${q.evidence_paragraph + 1}</code>` : "");
    wrapper.appendChild(meta);

    const heading = document.createElement("h3");
    heading.textContent = q.prompt;
    wrapper.appendChild(heading);

    const pair = document.createElement("div");
    pair.className = "pair";

    const left = document.createElement("div");
    if (q.answer_mode === "open") {
      const expected = document.createElement("p");
      expected.style.cssText = "margin:0 0 6px;font-size:13px";
      expected.innerHTML =
        `<span class="tag">written answer</span> ` +
        `<em>${escapeHtml(q.reference_answer || "")}</em>`;
      left.appendChild(expected);
    } else {
      const list = document.createElement("ol");
      for (const opt of q.options) {
        const li = document.createElement("li");
        li.textContent = opt.label;
        if (opt.is_correct) li.className = "right";
        list.appendChild(li);
      }
      left.appendChild(list);
    }
    if (q.explanation) {
      const why = document.createElement("p");
      why.className = "hint";
      why.style.margin = "0";
      why.textContent = q.explanation;
      left.appendChild(why);
    }

    const source = document.createElement("div");
    source.className = "pane";
    source.textContent = q.source_text || "No source recorded.";

    pair.append(left, source);
    wrapper.appendChild(pair);

    if (q.follow_up) {
      const fu = document.createElement("div");
      fu.style.cssText = "margin-top:10px;padding-left:12px;border-left:2px solid var(--hair)";
      const label = document.createElement("p");
      label.className = "eyebrow";
      label.style.margin = "0 0 4px";
      label.textContent = "Follow-up if they get it wrong";
      const prompt = document.createElement("p");
      prompt.style.cssText = "margin:0 0 6px;font-size:13px";
      prompt.textContent = q.follow_up.prompt;
      const opts = document.createElement("ol");
      for (const opt of q.follow_up.options) {
        const li = document.createElement("li");
        li.textContent = opt.label;
        if (opt.is_correct) li.className = "right";
        opts.appendChild(li);
      }
      fu.append(label, prompt, opts);
      wrapper.appendChild(fu);
    }

    const actions = document.createElement("div");
    actions.className = "actions";

    const approve = document.createElement("button");
    approve.textContent = q.status === "approved" ? "Approved" : "Approve";
    approve.disabled = q.status === "approved";
    approve.onclick = () => questionAction(q.id, "approve");

    const reject = document.createElement("button");
    reject.className = "quiet";
    reject.textContent = "Reject";
    reject.onclick = () => questionAction(q.id, "reject");

    const remove = document.createElement("button");
    remove.className = "quiet";
    remove.textContent = "Delete";
    remove.onclick = () => deleteQuestion(q.id);

    actions.append(approve, reject, remove);
    wrapper.appendChild(actions);
    host.appendChild(wrapper);
  }

  if (set.total) {
    say($("quizStatus"), `${set.approved} of ${set.total} approved.`);
  }
  $("coverage").textContent = set.paragraphs
    ? `${set.covered} of ${set.paragraphs} paragraphs have an approved question.`
    : "";
}

async function questionAction(id, verb) {
  try {
    await call(`/questions/${id}/${verb}`, { method: "POST" });
    await loadQuestions();
  } catch (err) {
    say($("quizStatus"), err.message, true);
  }
}

async function deleteQuestion(id) {
  try {
    await call(`/questions/${id}`, { method: "DELETE" });
    await loadQuestions();
  } catch (err) {
    say($("quizStatus"), err.message, true);
  }
}

loadStories().catch((err) => say($("uploadStatus"), err.message, true));
</script>
</body>
</html>
"""

@router.get("/authoring", response_class=HTMLResponse)
def authoring_console(request: Request):
    if not request.session.get("admin_user"):
        return RedirectResponse("/admin/login")
    return HTMLResponse(PAGE.replace("__API__", settings.api_prefix))
