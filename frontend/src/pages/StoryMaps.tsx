import { Link } from "react-router-dom";

import { GraphIcon, LockIcon, UsersIcon } from "../components/Icons";
import { EmptyState, ErrorState, LoadingCards } from "../components/States";
import { useLang } from "../lib/i18n";
import { useFetch } from "../lib/useFetch";
import { api, completedCount, isLocked, type Level } from "../api";

const LEVEL_KEY = {
  beginner: "levelBeginner",
  intermediate: "levelIntermediate",
  advanced: "levelAdvanced",
} as const satisfies Record<Level, "levelBeginner" | "levelIntermediate" | "levelAdvanced">;

export default function StoryMaps() {
  const { t, tx, has } = useLang();
  const { data, error, loading, reload } = useFetch(() => api.listStories());

  const stories = (data ?? []).filter((s) => has(s.title));
  const completed = completedCount(stories);

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl sm:text-4xl">{t("storyMapsTitle")}</h1>
        <p className="text-ink-muted">{t("storyMapsSubtitle")}</p>
      </header>

      {loading && <LoadingCards />}
      {!loading && error && <ErrorState error={error} onRetry={reload} />}
      {!loading && !error && stories.length === 0 && <EmptyState message={t("noStories")} />}

      {!loading && !error && stories.length > 0 && (
        <div className="mt-7 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {stories.map((story) => {
            const locked = isLocked(story, completed);

            const ready = story.character_count > 0 && !locked;

            return (
              <article
                key={story.id}
                className={`card flex flex-col gap-3 p-5 ${ready ? "" : "bg-canvas/60"}`}
              >
                <p className={ready ? "eyebrow" : "eyebrow-muted"}>{t(LEVEL_KEY[story.level])}</p>

                <div className="space-y-2">
                  <h2 className={`text-lg ${ready ? "text-ink" : "text-ink-muted"}`}>
                    {tx(story.title)}
                  </h2>
                  <p
                    className={`text-sm leading-relaxed ${
                      ready ? "text-ink-soft" : "text-ink-muted"
                    }`}
                  >
                    {tx(story.summary)}
                  </p>
                </div>

                <div className="mt-auto space-y-3 pt-1">
                  {locked ? (
                    <p className="flex items-center gap-1.5 text-xs text-ink-muted">
                      <LockIcon className="h-3.5 w-3.5" />
                      {t("unlockHint", story.unlock_after ?? 0)}
                    </p>
                  ) : ready ? (
                    <div className="flex items-center gap-4 text-xs text-ink-muted">
                      <span className="flex items-center gap-1.5">
                        <UsersIcon className="h-3.5 w-3.5" />
                        {story.character_count} {t("charactersCount")}
                      </span>
                      <span className="flex items-center gap-1.5">
                        <GraphIcon className="h-3.5 w-3.5" />
                        {t("storyMapTitle")}
                      </span>
                    </div>
                  ) : (
                    <p className="text-xs text-ink-muted">{t("mapNotReady")}</p>
                  )}

                  {ready ? (
                    <Link to={`/story/${story.id}/map`} className="btn-outline w-full">
                      {t("openStoryMap")}
                    </Link>
                  ) : (
                    <p className="btn-locked">
                      {locked ? t("locked") : t("openStoryMap")}
                    </p>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
