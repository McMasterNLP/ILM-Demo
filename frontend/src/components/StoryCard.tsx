import { Link } from "react-router-dom";

import ProgressBar from "./ProgressBar";
import { CheckCircleIcon, ClockIcon, LockIcon } from "./Icons";
import { useLang } from "../lib/i18n";
import { isLocked, percentOf, type Level, type Story } from "../api";

const LEVEL_KEY = {
  beginner: "levelBeginner",
  intermediate: "levelIntermediate",
  advanced: "levelAdvanced",
} as const satisfies Record<Level, "levelBeginner" | "levelIntermediate" | "levelAdvanced">;

export default function StoryCard({
  story,
  completed,
}: {
  story: Story;

  completed: number;
}) {
  const { t, tx } = useLang();

  const locked = isLocked(story, completed);
  const progress = story.progress;
  const percent = percentOf(progress, story.paragraph_count);
  const isDone = progress.completed;
  const started = !isDone && percent > 0;
  const hasQuiz = story.question_count > 0;
  const quizResult = progress.quiz;

  return (
    <article
      className={`card flex flex-col gap-3 p-5 ${locked ? "bg-canvas/60" : ""}`}
      aria-disabled={locked || undefined}
    >
      <div className="flex items-center justify-between gap-2">
        <p className={locked ? "eyebrow-muted" : "eyebrow"}>{t(LEVEL_KEY[story.level])}</p>
        {isDone && (
          <span className="flex items-center gap-1 text-[11px] font-medium text-brand">
            <CheckCircleIcon className="h-3.5 w-3.5" />
            {t("completed")}
          </span>
        )}
      </div>

      <div className="space-y-2">
        <h2 className={`text-lg ${locked ? "text-ink-muted" : "text-ink"}`}>{tx(story.title)}</h2>
        <p className={`text-sm leading-relaxed ${locked ? "text-ink-muted" : "text-ink-soft"}`}>
          {tx(story.summary)}
        </p>
      </div>

      <div className="mt-auto space-y-3 pt-1">
        {locked ? (
          <p className="flex items-center gap-1.5 text-xs text-ink-muted">
            <LockIcon className="h-3.5 w-3.5" />
            {t("unlockHint", story.unlock_after ?? 0)}
          </p>
        ) : (
          <div className="space-y-2">
            <div className="flex items-center gap-4 text-xs text-ink-muted">
              <span className="flex items-center gap-1.5">
                <ClockIcon className="h-3.5 w-3.5" />
                {story.minutes} {t("minutes")}
              </span>
            </div>
            {(started || isDone) && <ProgressBar value={percent} />}
            {quizResult && (
              <p className="flex items-center gap-1.5 text-xs text-ink-muted">
                <CheckCircleIcon className="h-3.5 w-3.5 text-brand" />
                <span className="font-medium text-brand">
                  {t("quizScoreShort", quizResult.correct, quizResult.total)}
                </span>
              </p>
            )}
          </div>
        )}

        {locked ? (
          <p className="btn-locked">{t("locked")}</p>
        ) : (
          <div className="space-y-2">
            <Link to={`/story/${story.id}`} className="btn-outline w-full">
              {isDone
                ? t("readAgain")
                : started
                  ? `${t("continueReading")} — ${percent}%`
                  : t("startReading")}
            </Link>
            {isDone && hasQuiz && (
              <Link to={`/quiz/${story.id}`} className="btn-quiet w-full">
                {progress.quiz_in_progress
                  ? t("resumeQuiz")
                  : quizResult
                    ? t("retakeQuiz")
                    : t("takeQuiz")}
              </Link>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
