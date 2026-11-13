import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import ProgressBar from "../components/ProgressBar";
import { ErrorState, LoadingPanel } from "../components/States";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  RefreshIcon,
  XCircleIcon,
} from "../components/Icons";
import { useLang, type StringKey } from "../lib/i18n";
import { useFetch } from "../lib/useFetch";
import {
  api,
  type AnswerResult,
  type Option,
  type Question,
  type QuestionType,
  type Quiz as QuizPayload,
  type StoryDetail,
} from "../api";

const TYPE_KEY = {
  relationship: "typeRelationship",
  identification: "typeIdentification",
  comprehension: "typeComprehension",
} as const satisfies Record<QuestionType, StringKey>;

function stableOptionOrder(options: Option[], seed: string): Option[] {
  function hash(value: string): number {
    let h = 2166136261;

    for (let i = 0; i < value.length; i += 1) {
      h ^= value.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }

    return h >>> 0;
  }

  return [...options].sort(
    (a, b) => hash(`${seed}:${a.id}`) - hash(`${seed}:${b.id}`)
  );
}

function OptionList({
  options,
  name,
  value,
  onChange,
  disabled,
}: {
  options: Option[];
  name: string;
  value: string | null;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  const { tx } = useLang();
  return (
    <div className="space-y-1">
      {options.map((opt) => (
        <label
          key={opt.id}
          className={`flex items-center gap-3 rounded-lg px-1 py-2 text-sm ${
            disabled ? "cursor-default text-ink-muted" : "cursor-pointer text-ink hover:bg-canvas"
          }`}
        >
          <input
            type="radio"
            name={name}
            value={opt.id}
            checked={value === opt.id}
            disabled={disabled}
            onChange={() => onChange(opt.id)}
            className="h-4 w-4 accent-brand"
          />
          {tx(opt.label)}
        </label>
      ))}
    </div>
  );
}

export default function Quiz() {
  const { storyId = "" } = useParams();
  const { t, tx, lang } = useLang();

  const quiz = useFetch<QuizPayload>(() => api.getQuiz(storyId, lang), [storyId, lang]);
  const story = useFetch<StoryDetail>(() => api.getStory(storyId), [storyId]);

  const [attemptId, setAttemptId] = useState<string | null>(null);
  const [attemptError, setAttemptError] = useState<Error | null>(null);
  const [index, setIndex] = useState(0);
  const [choice, setChoice] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResult | null>(null);
  const [followUpChoice, setFollowUpChoice] = useState<string | null>(null);
  const [followUpResult, setFollowUpResult] = useState<AnswerResult | null>(null);
  const [written, setWritten] = useState("");

  useEffect(() => {
    setAttemptId(null);
    setIndex(0);
    setChoice(null);
    setWritten("");
    setResult(null);
    setFollowUpChoice(null);
    setFollowUpResult(null);
    setFinished(null);
  }, [lang]);
  const [submitting, setSubmitting] = useState(false);
  const [finished, setFinished] = useState<{ score: number; total: number } | null>(null);

  useEffect(() => {
    const data = quiz.data;
    if (!data || attemptId) return;
    let cancelled = false;
    api
      .startAttempt(
        storyId,
        lang,
        data.questions.map((q) => q.id)
      )
      .then((attempt) => {
        if (cancelled) return;
        setAttemptId(attempt.id);
        const answered = new Set(attempt.answered_question_ids);
        const resumeIndex = data.questions.findIndex((q) => !answered.has(q.id));
        if (resumeIndex > 0) setIndex(resumeIndex);
      })
      .catch((err: unknown) => {
        if (!cancelled) setAttemptError(err instanceof Error ? err : new Error("Failed"));
      });
    return () => {
      cancelled = true;
    };
  }, [quiz.data, attemptId, storyId, lang]);

  if (quiz.loading || story.loading) {
    return (
      <div className="mx-auto w-full max-w-3xl px-6 py-10">
        <LoadingPanel />
      </div>
    );
  }

  const failure = quiz.error ?? attemptError;
  if (failure || !quiz.data || quiz.data.questions.length === 0) {
    return (
      <div className="mx-auto w-full max-w-3xl px-6 py-10">
        <ErrorState
          error={failure ?? new Error(t("noQuestionsInLanguage"))}
          onRetry={quiz.reload}
        />
        <div className="mt-4 flex gap-3">
          <Link to="/quiz" className="btn-quiet">
            {t("quizzesTitle")}
          </Link>
          <Link to={`/story/${storyId}`} className="btn-outline">
            {t("readTheStory")}
          </Link>
        </div>
      </div>
    );
  }

  const questions = quiz.data.questions;
  const question: Question = questions[index];
  const isLastQuestion = index >= questions.length - 1;
  const Arrow = lang === "ar" ? ArrowLeftIcon : ArrowRightIcon;
  const followUp = result?.follow_up ?? null;

  const isOpen = question.answer_mode === "open";
  const neutralGrading =
    result?.grading_status === "not_enough_evidence" ||
    result?.grading_status === "grading_unavailable";

  async function submit() {
    if (!attemptId) return;
    if (isOpen ? !written.trim() : !choice) return;

    setSubmitting(true);
    try {
      setResult(
        await api.answer(attemptId, {
          question_id: question.id,
          ...(isOpen ? { text: written, lang } : { option_id: choice ?? undefined }),
        })
      );
    } catch (err) {
      setAttemptError(err instanceof Error ? err : new Error("Failed"));
    } finally {
      setSubmitting(false);
    }
  }

  async function submitFollowUp() {
    if (!attemptId || !followUp || !followUpChoice) return;
    setSubmitting(true);
    try {
      setFollowUpResult(
        await api.answer(attemptId, {
          question_id: followUp.id,
          option_id: followUpChoice,
          was_followup: true,
        })
      );
    } catch (err) {
      setAttemptError(err instanceof Error ? err : new Error("Failed"));
    } finally {
      setSubmitting(false);
    }
  }

  async function advance() {
    if (index >= questions.length - 1) {
      const attempt = attemptId ? await api.finishAttempt(attemptId).catch(() => null) : null;
      setFinished({
        score: attempt?.score ?? result?.score ?? 0,
        total: attempt?.total ?? questions.length,
      });
      return;
    }
    setIndex((i) => i + 1);
    setChoice(null);
    setWritten("");
    setResult(null);
    setFollowUpChoice(null);
    setFollowUpResult(null);
  }

  function restart() {

    setAttemptId(null);
    setIndex(0);
    setChoice(null);
    setWritten("");
    setResult(null);
    setFollowUpChoice(null);
    setFollowUpResult(null);
    setFinished(null);
  }

  if (finished) {
    return (
      <div className="mx-auto w-full max-w-xl px-6 py-16 text-center">
        <div className="card p-8">
          <CheckCircleIcon className="mx-auto h-8 w-8 text-brand" />
          <h1 className="mt-3 text-2xl">{t("quizDone")}</h1>
          <p className="mt-2 text-sm text-ink-soft">
            {t("quizScore", finished.score, finished.total)}
          </p>
          <div className="mt-6 flex justify-center gap-3">
            <button type="button" className="btn-quiet" onClick={restart}>
              <RefreshIcon className="h-4 w-4" />
              {t("restartQuiz")}
            </button>
            <Link to="/progress" className="btn-outline">
              {t("navProgress")}
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const correctLabel = result
    ? question.options.find((o) => o.id === result.correct_option_id)?.label
    : undefined;
  const chosenLabel = question.options.find((o) => o.id === choice)?.label;

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-10">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex-1">
          <p className="text-sm text-ink-muted">
            {t("questionOf", index + 1, questions.length)}
            {story.data ? ` — ${tx(story.data.title)}` : ""}
          </p>
          <ProgressBar value={((index + 1) / questions.length) * 100} className="mt-2 max-w-sm" />
        </div>
        <span className="rounded-md border border-brand px-2.5 py-1 text-xs text-brand">
          {t("questionTypeLabel", t(TYPE_KEY[question.type]))}
        </span>
      </div>

      <div className="card mt-8 p-7">
        <p className="eyebrow">{t("questionLabel", index + 1)}</p>
        <h1 className="mt-3 text-2xl leading-snug">{tx(question.prompt)}</h1>

        <div className="mt-5">
          {isOpen ? (
            <>
              <label htmlFor="written" className="text-xs text-ink-soft">
                {t("writeAnswer")}
              </label>
              <textarea
                id="written"
                value={written}
                onChange={(e) => setWritten(e.target.value)}
                disabled={Boolean(result)}
                rows={4}
                placeholder={t("answerPlaceholder")}
                className="mt-1.5 w-full rounded-lg border border-hair bg-white px-3 py-2 text-sm leading-relaxed focus:border-brand focus:outline-none disabled:bg-canvas disabled:text-ink-muted"
              />
            </>
          ) : (
            <OptionList
              options={stableOptionOrder(question.options, question.id)}
              name={question.id}
              value={choice}
              onChange={setChoice}
              disabled={Boolean(result)}
            />
          )}
        </div>

        {!result && (
          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              type="button"
              className="btn-outline flex-1"
              disabled={(isOpen ? !written.trim() : !choice) || submitting || !attemptId}
              onClick={submit}
            >
              {submitting && isOpen ? t("checking") : t("confirmAnswer")}
            </button>
            <button type="button" className="text-sm text-brand hover:underline" onClick={advance}>
              {t("skipQuestion")}
            </button>
          </div>
        )}
      </div>

      {result && (
        <div className="card mt-4 p-7">
          <p className="flex items-center gap-2 text-sm font-medium">
            {neutralGrading ? (
              <RefreshIcon className="h-5 w-5 text-brand" />
            ) : result.is_correct ? (
              <CheckCircleIcon className="h-5 w-5 text-brand" />
            ) : (
              <XCircleIcon className="h-5 w-5 text-brand" />
            )}
            {result.grading_status === "not_enough_evidence"
              ? lang === "ar"
                ? "لا توجد أدلة معتمدة كافية"
                : "Not enough approved evidence"
              : result.grading_status === "grading_unavailable"
                ? lang === "ar"
                  ? "التقييم غير متاح مؤقتًا"
                  : "Grading temporarily unavailable"
                : result.is_correct
                  ? t("correctHeading")
                  : t("incorrectHeading")}
          </p>

          {result.feedback && (
            <p className="mt-3 text-sm leading-relaxed text-ink">{result.feedback}</p>
          )}

          {neutralGrading && (
            <button
              type="button"
              className="btn-quiet mt-4"
              onClick={() => setResult(null)}
            >
              {lang === "ar"
                ? "تعديل الإجابة والمحاولة مرة أخرى"
                : "Edit answer and try again"}
            </button>
          )}

          {!result.is_correct &&
            !neutralGrading &&
            isOpen &&
            result.reference_answer && (
            <div className="mt-4 text-sm">
              <p className="text-ink-muted">
                {t("yourAnswer")} {written || "—"}
              </p>
              <p className="mt-1">
                {t("expectedAnswer")}{" "}
                <span className="font-medium text-brand">{result.reference_answer}</span>
              </p>
            </div>
          )}

          {!result.is_correct && !isOpen && (
            <div className="mt-4 space-y-1 text-sm">
              <p className="text-ink-muted">
                {t("yourAnswer")} {chosenLabel ? tx(chosenLabel) : "—"}
              </p>
              <p>
                {t("correctAnswer")}{" "}
                <span className="font-medium text-brand">
                  {correctLabel ? tx(correctLabel) : "—"}
                </span>
              </p>
            </div>
          )}

          {result.source_text && (
            <div className="mt-4 rounded-lg border border-hair bg-stone-50 p-4">
              <p className="eyebrow-muted">{t("sourcePassage")}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">
                {result.source_text}
              </p>
            </div>
          )}

          {result.kg_edge && (
            <pre
              className="mt-4 overflow-x-auto rounded-lg bg-stone-100 px-4 py-3 font-mono text-xs text-ink-soft"
              dir="ltr"
            >
              {result.kg_edge}
            </pre>
          )}

          <p className="mt-4 text-sm leading-relaxed text-ink-soft">{tx(result.explanation)}</p>
        </div>
      )}

      {result && !result.is_correct && !neutralGrading && followUp && (
        <>
          <p className="mt-6 text-sm text-ink-muted">{t("tryAgainDifferent")}</p>
          <div className="card mt-2 p-7">
            <p className="eyebrow">{t("followUpQuestion")}</p>
            <h2 className="mt-3 text-xl leading-snug">{tx(followUp.prompt)}</h2>
            <div className="mt-5">
              <OptionList
                options={stableOptionOrder(followUp.options, followUp.id)}
                name={`${followUp.id}-followup`}
                value={followUpChoice}
                onChange={setFollowUpChoice}
                disabled={Boolean(followUpResult)}
              />
            </div>

            {followUpResult && (
              <p className="mt-3 flex items-center gap-2 text-sm">
                {followUpResult.is_correct ? (
                  <CheckCircleIcon className="h-4 w-4 text-brand" />
                ) : (
                  <XCircleIcon className="h-4 w-4 text-brand" />
                )}
                {followUpResult.is_correct ? t("correctHeading") : t("incorrectHeading")}
              </p>
            )}

            <div className="mt-6 flex flex-wrap items-center gap-3">
              <Link
                to={`/story/${storyId}${
                  question.evidence_paragraph !== null ? `?p=${question.evidence_paragraph}` : ""
                }`}
                className="btn-quiet"
              >
                <RefreshIcon className="h-4 w-4" />
                {t("rereadParagraph")}
              </Link>
              {!followUpResult ? (
                <button
                  type="button"
                  className="btn-outline flex-1"
                  disabled={!followUpChoice || submitting}
                  onClick={submitFollowUp}
                >
                  {t("confirmAnswer")}
                </button>
              ) : (
                <button type="button" className="btn-outline flex-1" onClick={advance}>
                  {isLastQuestion ? t("seeResults") : t("nextQuestion")}
                  <Arrow className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        </>
      )}

      {result && (result.is_correct || !followUp) && (
        <div className="mt-6 flex justify-end">
          <button type="button" className="btn-outline" onClick={advance}>
            {isLastQuestion ? t("seeResults") : t("nextQuestion")}
            <Arrow className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}
