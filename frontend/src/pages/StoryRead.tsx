import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import Breadcrumb from "../components/Breadcrumb";
import ProgressBar from "../components/ProgressBar";
import { ErrorState, LoadingPanel } from "../components/States";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  GraphIcon,
  RefreshIcon,
} from "../components/Icons";
import { useLang } from "../lib/i18n";
import { useFetch } from "../lib/useFetch";
import { api, type StoryDetail, type StoryGraphData } from "../api";

export default function StoryRead() {
  const { storyId = "" } = useParams();
  const [params] = useSearchParams();
  const { t, tx, lang, has } = useLang();

  const story = useFetch<StoryDetail>(() => api.getStory(storyId), [storyId]);
  const graph = useFetch<StoryGraphData | null>(
    () => api.getGraph(storyId).catch(() => null),
    [storyId]
  );

  const [current, setCurrent] = useState<number | null>(null);
  const [completed, setCompleted] = useState(false);

  const total = story.data?.paragraphs.length ?? 0;

  useEffect(() => {
    if (!story.data || current !== null) return;
    const raw = params.get("p");
    const requested = raw === null ? null : Number(raw);
    const resume =
      requested !== null && Number.isInteger(requested) && requested >= 0
        ? requested
        : story.data.progress.paragraph;
    setCurrent(total > 0 ? Math.min(resume, total - 1) : 0);
    setCompleted(story.data.progress.completed);
  }, [story.data, params, current, total]);

  useEffect(() => {
    if (current === null || total === 0) return;
    let cancelled = false;
    api
      .setParagraph(storyId, current)
      .then((entry) => {
        if (!cancelled) setCompleted(entry.completed);
      })
      .catch(() => {

      });
    return () => {
      cancelled = true;
    };
  }, [storyId, current, total]);

  const characters = useMemo(() => {
    if (!story.data || !graph.data || current === null) return [];
    const keys = story.data.paragraphs[current]?.characters ?? [];
    return keys
      .map((key) => graph.data!.nodes.find((n) => n.id === key))
      .filter((n): n is NonNullable<typeof n> => Boolean(n));
  }, [story.data, graph.data, current]);

  if (story.loading || current === null) {
    return <div className="mx-auto w-full max-w-6xl px-6 py-8">{<LoadingPanel />}</div>;
  }

  if (story.error || !story.data) {
    return (
      <div className="mx-auto w-full max-w-6xl px-6 py-8">
        <ErrorState
          error={story.error ?? new Error(t("notFound"))}
          onRetry={story.reload}
        />
      </div>
    );
  }

  const detail = story.data;
  const paragraph = detail.paragraphs[current];
  const atEnd = current >= total - 1;
  const Arrow = lang === "ar" ? ArrowLeftIcon : ArrowRightIcon;
  const BackArrow = lang === "ar" ? ArrowRightIcon : ArrowLeftIcon;

  async function readAgain() {
    await api.restartStory(storyId).catch(() => null);
    setCompleted(false);
    setCurrent(0);
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-8">
      <div className="overflow-hidden rounded-xl border border-hair">
        <div className="grid lg:grid-cols-[1fr_320px]">
          <div className="flex flex-col p-7">
            <Breadcrumb
              items={[
                { label: t("breadcrumbStories"), to: "/" },
                { label: tx(detail.title) },
              ]}
            />

            <h1 className="mt-3 text-2xl sm:text-3xl">{tx(detail.heading)}</h1>

            <div className="mt-6 flex-1 space-y-5">
              {paragraph && (() => {
                const missing = paragraph.pending || !has(paragraph.text);
                return (
                  <p
                    key={paragraph.position}
                    className={`leading-8 ${
                      missing ? "italic text-ink-muted" : "text-ink-soft"
                    }`}
                  >
                    {missing ? t("paragraphPending") : tx(paragraph.text)}
                  </p>
                );
              })()}

              {atEnd && completed && (
                <div className="rounded-xl border border-brand/25 bg-brand-tint p-5">
                  <p className="flex items-center gap-2 text-sm font-medium text-brand">
                    <CheckCircleIcon className="h-5 w-5" />
                    {t("storyFinishedTitle")}
                  </p>
                  <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">
                    {t("storyFinishedBody")}
                  </p>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <Link to={`/story/${detail.id}/map`} className="btn-outline bg-white">
                      <GraphIcon className="h-4 w-4" />
                      {t("openStoryMap")}
                    </Link>
                    {detail.question_count > 0 && (
                      <Link to={`/quiz/${detail.id}`} className="btn-outline bg-white">
                        {t("takeQuiz")}
                        <Arrow className="h-4 w-4" />
                      </Link>
                    )}
                    <button type="button" className="btn-quiet" onClick={readAgain}>
                      <RefreshIcon className="h-4 w-4" />
                      {t("readAgain")}
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-8 border-t border-hair pt-5">
              <div className="flex flex-wrap items-center gap-4">
                <span className="text-xs text-ink-muted">
                  {t("paragraphOf", current + 1, total)}
                </span>
                <ProgressBar
                  value={((current + 1) / Math.max(total, 1)) * 100}
                  className="min-w-[8rem] flex-1"
                />
                {current > 0 && (
                  <button
                    type="button"
                    className="btn-quiet"
                    onClick={() => setCurrent((c) => (c ?? 0) - 1)}
                  >
                    <BackArrow className="h-4 w-4" />
                    {t("previous")}
                  </button>
                )}
                {!atEnd && (
                  <button
                    type="button"
                    className="btn-outline"
                    onClick={() => setCurrent((c) => (c ?? 0) + 1)}
                  >
                    {t("next")}
                    <Arrow className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>
          </div>

          <aside className="border-hair bg-white p-6 lg:border-s">
            <p className="eyebrow-muted">{t("charactersInSection")}</p>

            <div className="mt-3 flex flex-wrap gap-2">
              {characters.filter((node) => has(node.label)).map((node, i) => (
                <Link
                  key={node.id}
                  to={`/story/${detail.id}/map?node=${node.id}`}
                  className={`rounded-md px-2.5 py-1 text-xs text-white transition-opacity hover:opacity-90 ${
                    i === 0 ? "bg-brand-deep" : "bg-stone-600"
                  }`}
                >
                  {tx(node.label)}
                </Link>
              ))}
            </div>

            <div className="card mt-5 p-4">
              <GraphIcon className="h-5 w-5 text-brand" />
              <h2 className="mt-2 text-sm font-medium">{t("discoverRelationships")}</h2>
              <p className="mt-1 text-xs leading-relaxed text-ink-soft">{t("discoverBody")}</p>
              <Link to={`/story/${detail.id}/map`} className="btn-outline mt-4 w-full">
                {t("openStoryMap")}
              </Link>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
