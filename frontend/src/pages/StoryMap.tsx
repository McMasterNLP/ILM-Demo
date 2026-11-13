import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import Breadcrumb from "../components/Breadcrumb";
import StoryGraph from "../components/StoryGraph";
import { CloseIcon, PersonIcon, PinIcon } from "../components/Icons";
import { ErrorState, LoadingPanel } from "../components/States";
import { useLang } from "../lib/i18n";
import { useFetch } from "../lib/useFetch";
import { api, type StoryDetail, type StoryGraphData } from "../api";

const KEEP_OPEN = "[data-node], [data-graph-control], [data-story-card]";

export default function StoryMap() {
  const { storyId = "" } = useParams();
  const [params] = useSearchParams();
  const { t, tx, has, lang } = useLang();

  const story = useFetch<StoryDetail>(() => api.getStory(storyId), [storyId]);
  const graph = useFetch<StoryGraphData>(() => api.getGraph(storyId), [storyId]);

  const [selectedId, setSelectedId] = useState<string | null>(params.get("node"));
  const [mapMode, setMapMode] = useState<"simple" | "detailed">("simple");
  const cardRef = useRef<HTMLElement | null>(null);

  const visible = useMemo(() => {
    if (!graph.data) return null;

    const provenanceAware = graph.data.edges.some(
      (edge) => edge.layer === "automatic" || edge.layer === "quran"
    );

    let edges = graph.data.edges;

    if (provenanceAware) {
      edges =
        mapMode === "simple"
          ? graph.data.edges.filter((edge) => edge.layer === "automatic")
          : graph.data.edges.filter(
              (edge) => edge.layer === "automatic" || edge.layer === "quran"
            );
    }

    const endpointIds = new Set<string>();

    for (const edge of edges) {
      endpointIds.add(edge.from);
      endpointIds.add(edge.to);
    }

    const nodes = graph.data.nodes.filter(
      (node) => endpointIds.has(node.id) && has(node.label)
    );

    const ids = new Set(nodes.map((node) => node.id));

    return {
      nodes,
      edges: edges.filter((edge) => ids.has(edge.from) && ids.has(edge.to)),
    };
  }, [graph.data, has, mapMode]);

  const selected = useMemo(
    () => visible?.nodes.find((n) => n.id === selectedId) ?? null,
    [visible, selectedId]
  );

  const relationships = useMemo(() => {
    if (!visible || !selected) return [];
    const outgoing = visible.edges
      .filter((e) => e.from === selected.id)
      .map((e) => tx(e.phrase));
    const incoming = visible.edges
      .filter((e) => e.to === selected.id)
      .map((e) => {
        const other = visible.nodes.find((n) => n.id === e.from);
        return other ? `${tx(other.label)} → ${tx(e.label)}` : null;
      })
      .filter((s): s is string => Boolean(s));
    return [...outgoing, ...incoming];
  }, [visible, selected, tx]);

  useEffect(() => {
    if (!selectedId) return;

    const onPointerDown = (e: PointerEvent) => {
      const target = e.target;
      if (target instanceof Element && target.closest(KEEP_OPEN)) return;
      setSelectedId(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelectedId(null);
    };

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [selectedId]);

  if (story.loading || graph.loading) {
    return <div className="mx-auto w-full max-w-6xl px-6 py-8">{<LoadingPanel />}</div>;
  }

  const failure = story.error ?? graph.error;
  if (failure || !story.data || !graph.data) {
    return (
      <div className="mx-auto w-full max-w-6xl px-6 py-8">
        <ErrorState error={failure ?? new Error(t("notFound"))} onRetry={graph.reload} />
        <Link to="/map" className="btn-outline mt-4">
          {t("storyMapsTitle")}
        </Link>
      </div>
    );
  }

  const detail = story.data;

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Breadcrumb
            items={[
              { label: t("breadcrumbStories"), to: "/" },
              { label: tx(detail.title), to: `/story/${detail.id}` },
            ]}
          />
          <h1 className="mt-3 text-2xl sm:text-3xl">
            {t("storyMapTitle")} — {tx(detail.title)}
          </h1>
          <p className="mt-1 text-sm text-ink-muted">{t("storyMapSubtitle")}</p>
          <p className="mt-1 text-sm text-ink-muted">
            {lang === "ar"
              ? "استخدم العرض البسيط لمتابعة الأحداث الرئيسية، أو العرض المفصل لاستكشاف الروابط المدعومة من القرآن."
              : "Use Simple view for the main story journey, or Detailed view to explore additional Quran-supported connections."}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-1 py-1 text-xs text-ink-soft">
            <PersonIcon className="h-3.5 w-3.5" />
            {t("legendPerson")}
          </span>
          <span className="inline-flex items-center gap-1.5 px-1 py-1 text-xs text-ink-soft">
            <PinIcon className="h-3.5 w-3.5" />
            {t("legendPlace")}
          </span>
        </div>
      </div>

      <div
        data-graph-control
        className="mt-5 inline-flex rounded-xl border border-stone-200 bg-white p-1 shadow-sm"
      >
        <button
          type="button"
          onClick={() => {
            setMapMode("simple");
            setSelectedId(null);
          }}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            mapMode === "simple"
              ? "bg-brand text-white"
              : "text-ink-soft hover:bg-stone-50 hover:text-ink"
          }`}
        >
          {lang === "ar" ? "بسيط" : "Simple"}
        </button>

        <button
          type="button"
          onClick={() => {
            setMapMode("detailed");
            setSelectedId(null);
          }}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            mapMode === "detailed"
              ? "bg-brand text-white"
              : "text-ink-soft hover:bg-stone-50 hover:text-ink"
          }`}
        >
          {lang === "ar" ? "مفصل" : "Detailed"}
        </button>
      </div>

      <p className="mt-2 text-xs text-ink-muted">
        {mapMode === "simple"
          ? lang === "ar"
            ? `${visible?.edges.length ?? 0} روابط أساسية`
            : `${visible?.edges.length ?? 0} core story connections`
          : lang === "ar"
            ? `${visible?.edges.length ?? 0} روابط مع إثراء قرآني`
            : `${visible?.edges.length ?? 0} connections including Quran enrichment`}
      </p>

      <div className="relative mt-4">
        <StoryGraph
          nodes={visible?.nodes ?? []}
          edges={visible?.edges ?? []}
          selectedId={selectedId}
          onSelect={setSelectedId}
        />

        {selected && (
          <aside
            ref={cardRef}
            data-story-card
            className="card relative mt-4 w-full p-4 lg:absolute lg:top-1/3 lg:mt-0 lg:w-64 lg:shadow-pop lg:start-0"
          >
            <button
              type="button"
              onClick={() => setSelectedId(null)}
              aria-label={t("close")}
              title={t("close")}
              className="absolute top-2.5 end-2.5 rounded p-1 text-ink-muted transition-colors hover:bg-canvas hover:text-ink"
            >
              <CloseIcon className="h-3.5 w-3.5" />
            </button>

            <p className="eyebrow pe-6">{tx(selected.category)}</p>
            <h2 className="mt-1.5 text-base">{tx(selected.label)}</h2>
            <p className="mt-2 text-xs leading-relaxed text-ink-soft">{tx(selected.description)}</p>

            {relationships.length > 0 && (
              <>
                <p className="mt-4 text-xs text-ink-muted">{t("relationships")}</p>
                <ul className="mt-1.5 space-y-1.5">
                  {relationships.map((rel, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-ink-soft">
                      <span className="text-brand" aria-hidden>
                        →
                      </span>
                      {rel}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </aside>
        )}
      </div>
    </div>
  );
}
