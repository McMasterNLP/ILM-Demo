import { useState } from "react";

import StoryCard from "../components/StoryCard";
import { EmptyState, ErrorState, LoadingCards } from "../components/States";
import { useLang } from "../lib/i18n";
import { useFetch } from "../lib/useFetch";
import { api, completedCount, type Level } from "../api";

type Filter = "all" | Level;

const FILTERS: {
  id: Filter;
  key: "filterAll" | "levelBeginner" | "levelIntermediate" | "levelAdvanced";
}[] = [
  { id: "all", key: "filterAll" },
  { id: "beginner", key: "levelBeginner" },
  { id: "intermediate", key: "levelIntermediate" },
  { id: "advanced", key: "levelAdvanced" },
];

export default function Stories() {
  const { t, has } = useLang();
  const [filter, setFilter] = useState<Filter>("all");
  const { data, error, loading, reload } = useFetch(() => api.listStories());

  const stories = (data ?? []).filter((s) => has(s.title));
  const visible = filter === "all" ? stories : stories.filter((s) => s.level === filter);
  const completed = completedCount(stories);

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl sm:text-4xl">{t("storiesTitle")}</h1>
        <p className="text-ink-muted">{t("storiesSubtitle")}</p>
      </header>

      <div className="mt-7 inline-flex overflow-hidden rounded-lg border border-hair bg-white">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setFilter(f.id)}
            aria-pressed={filter === f.id}
            className={`border-e border-hair px-4 py-2 text-sm transition-colors last:border-e-0 ${
              filter === f.id
                ? "bg-brand-tint font-medium text-brand"
                : "text-ink-soft hover:bg-canvas"
            }`}
          >
            {t(f.key)}
          </button>
        ))}
      </div>

      {loading && <LoadingCards />}
      {!loading && error && <ErrorState error={error} onRetry={reload} />}
      {!loading && !error && visible.length === 0 && <EmptyState message={t("noStories")} />}

      {!loading && !error && visible.length > 0 && (
        <div className="mt-6 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {visible.map((story) => (
            <StoryCard key={story.id} story={story} completed={completed} />
          ))}
        </div>
      )}
    </div>
  );
}
