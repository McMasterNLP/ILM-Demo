import { Link } from "react-router-dom";

import ProgressBar from "../components/ProgressBar";
import { BookIcon, CheckCircleIcon, GraphIcon } from "../components/Icons";
import { ErrorState, LoadingPanel } from "../components/States";
import { useLang, type StringKey } from "../lib/i18n";
import { useFetch } from "../lib/useFetch";
import { api, percentOf, type LearnerSummary, type Story } from "../api";

function StatTile({
  icon,
  labelKey,
  value,
}: {
  icon: React.ReactNode;
  labelKey: StringKey;
  value: number;
}) {
  const { t } = useLang();
  return (
    <div className="card p-5">
      <span className="text-brand">{icon}</span>
      <p className="mt-3 text-3xl">{value}</p>
      <p className="mt-1 text-xs text-ink-muted">{t(labelKey)}</p>
    </div>
  );
}

export default function Progress() {
  const { t, tx, has } = useLang();

  const summary = useFetch<LearnerSummary>(() => api.getProgress());
  const stories = useFetch<Story[]>(() => api.listStories());

  if (summary.loading || stories.loading) {
    return (
      <div className="mx-auto w-full max-w-5xl px-6 py-10">
        <LoadingPanel />
      </div>
    );
  }

  const failure = summary.error ?? stories.error;
  if (failure || !summary.data || !stories.data) {
    return (
      <div className="mx-auto w-full max-w-5xl px-6 py-10">
        <ErrorState error={failure ?? new Error("No data")} onRetry={summary.reload} />
      </div>
    );
  }

  const byId = new Map(stories.data.map((story) => [story.id, story]));
  const rows = summary.data.progress
    .map((entry) => {
      const story = byId.get(entry.story_id);
      if (!story) return null;
      return {
        story,
        percent: percentOf(
          {
            paragraph: entry.paragraph,
            completed: entry.completed,
          },
          story.paragraph_count
        ),
        completed: entry.completed,
        quiz: entry.quiz,
      };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null)
    .filter((row) => row.percent > 0 && has(row.story.title));

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl">{t("progressTitle")}</h1>
        <p className="text-ink-muted">{t("progressSubtitle")}</p>
      </header>

      <div className="mt-7 grid gap-4 sm:grid-cols-3">
        <StatTile
          icon={<BookIcon className="h-5 w-5" />}
          labelKey="storiesStarted"
          value={summary.data.stories_started}
        />
        <StatTile
          icon={<CheckCircleIcon className="h-5 w-5" />}
          labelKey="storiesCompleted"
          value={summary.data.stories_completed}
        />
        <StatTile
          icon={<GraphIcon className="h-5 w-5" />}
          labelKey="questionsAnswered"
          value={summary.data.questions_answered}
        />
      </div>

      <section className="mt-8">
        <h2 className="eyebrow-muted">{t("readingProgress")}</h2>
        <ul className="card mt-2 divide-y divide-hair">
          {rows.length === 0 && (
            <li className="px-5 py-4 text-sm text-ink-muted">{t("nothingStartedYet")}</li>
          )}
          {rows.map(({ story, percent, completed, quiz }) => (
            <li key={story.id} className="flex flex-wrap items-center gap-4 px-5 py-4">
              <Link
                to={`/story/${story.id}`}
                className="flex min-w-40 flex-1 items-center gap-2 text-sm hover:text-brand"
              >
                {tx(story.title)}
                {completed && (
                  <span className="flex items-center gap-1 text-[11px] font-medium text-brand">
                    <CheckCircleIcon className="h-3.5 w-3.5" />
                    {t("completed")}
                  </span>
                )}
              </Link>
              {quiz && (
                <span className="text-xs font-medium text-brand">
                  {t("quizScoreShort", quiz.correct, quiz.total)}
                </span>
              )}
              <ProgressBar value={percent} className="min-w-[8rem] flex-1" />
              <span className="w-10 text-end text-xs text-ink-muted">{percent}%</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
