import { useEffect, useMemo, useRef, useState } from "react";

import { useLang } from "../lib/i18n";
import type { GraphEdge, GraphNode } from "../api";

const VIEW_W = 1400;
const VIEW_H = 760;

const NODE_H = 58;
const SELECTED_H = 64;
const BOUNDS_PAD = 28;
const DRAG_SLOP = 4;

type Point = { x: number; y: number };
type Size = { w: number; h: number };

type RoutedEdge = {
  edge: GraphEdge;
  index: number;
  parallelIndex: number;
  parallelCount: number;
};

function nodeWidth(label: string, hasIcon: boolean, arabic: boolean, important: boolean) {
  const per = arabic ? 10.4 : 8.8;
  const base = Math.round(label.length * per) + 54 + (hasIcon ? 22 : 0);
  return Math.max(important ? 150 : 128, Math.min(base, 300));
}

function pairKey(a: string, b: string) {
  return [a, b].sort().join("::");
}

function isSymmetric(label: string) {
  const normalized = label.toLowerCase();
  return (
    normalized.includes("sibling") ||
    normalized.includes("reunited") ||
    label.includes("\u0623\u062e") ||
    label.includes("\u0623\u062e\u062a") ||
    label.includes("\u0627\u062c\u062a\u0645\u0639")
  );
}

function boundaryPoint(from: Point, to: Point, size: Size): Point {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const absDx = Math.abs(dx);
  const absDy = Math.abs(dy);

  if (absDx === 0 && absDy === 0) return to;

  const scaleX = absDx > 0 ? size.w / 2 / absDx : Number.POSITIVE_INFINITY;
  const scaleY = absDy > 0 ? size.h / 2 / absDy : Number.POSITIVE_INFINITY;
  const scale = Math.min(scaleX, scaleY);

  return {
    x: to.x - dx * scale,
    y: to.y - dy * scale,
  };
}

function buildInitialLayout(nodes: GraphNode[], edges: GraphEdge[]): Map<string, Point> {
  const result = new Map<string, Point>();
  if (!nodes.length) return result;

  const degree = new Map(nodes.map((node) => [node.id, 0]));
  const adjacency = new Map(nodes.map((node) => [node.id, [] as string[]]));

  for (const edge of edges) {
    degree.set(edge.from, (degree.get(edge.from) ?? 0) + 1);
    degree.set(edge.to, (degree.get(edge.to) ?? 0) + 1);

    adjacency.get(edge.from)?.push(edge.to);
    adjacency.get(edge.to)?.push(edge.from);
  }

  const ranked = [...nodes].sort((a, b) => {
    const d = (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0);
    if (d !== 0) return d;
    return a.id.localeCompare(b.id);
  });

  const hub = ranked[0];
  const center = { x: VIEW_W / 2, y: VIEW_H / 2 };

  result.set(hub.id, center);

  /*
   * A peripheral leaf whose only neighbour is itself away from the main
   * hub belongs visually with that neighbour. For example, if a family
   * node connects only to a place, it should appear near that place rather
   * than being independently distributed around the central character.
   */
  const satellites = ranked.filter((node) => {
    if (node.id === hub.id) return false;
    if ((degree.get(node.id) ?? 0) !== 1) return false;

    const neighbour = adjacency.get(node.id)?.[0];
    return Boolean(neighbour && neighbour !== hub.id);
  });

  const satelliteIds = new Set(satellites.map((node) => node.id));

  const primary = ranked.filter(
    (node) =>
      node.id !== hub.id &&
      (degree.get(node.id) ?? 0) > 0 &&
      !satelliteIds.has(node.id)
  );

  const isolated = ranked.filter((node) => (degree.get(node.id) ?? 0) === 0);

  const radiusX = Math.min(470, 280 + primary.length * 24);
  const radiusY = Math.min(275, 185 + primary.length * 11);

  primary.forEach((node, index) => {
    const angle =
      -Math.PI / 2 + (index * Math.PI * 2) / Math.max(primary.length, 1);

    result.set(node.id, {
      x: center.x + Math.cos(angle) * radiusX,
      y: center.y + Math.sin(angle) * radiusY,
    });
  });

  const satellitesByParent = new Map<string, GraphNode[]>();

  for (const node of satellites) {
    const parent = adjacency.get(node.id)?.[0];
    if (!parent) continue;

    const group = satellitesByParent.get(parent) ?? [];
    group.push(node);
    satellitesByParent.set(parent, group);
  }

  for (const [parentId, children] of satellitesByParent) {
    const parent = result.get(parentId) ?? center;

    let dx = parent.x - center.x;
    let dy = parent.y - center.y;
    let length = Math.hypot(dx, dy);

    if (length < 1) {
      dx = 1;
      dy = 0;
      length = 1;
    }

    const ux = dx / length;
    const uy = dy / length;

    const px = -uy;
    const py = ux;

    children.forEach((node, index) => {
      const spread = (index - (children.length - 1) / 2) * 130;

      let x = parent.x + ux * 175 + px * spread;
      let y = parent.y + uy * 135 + py * spread;

      x = Math.max(115, Math.min(VIEW_W - 115, x));
      y = Math.max(80, Math.min(VIEW_H - 80, y));

      result.set(node.id, { x, y });
    });
  }

  isolated.forEach((node, index) => {
    const columns = Math.max(1, Math.min(3, isolated.length));
    const col = index % columns;
    const row = Math.floor(index / columns);
    const spacing = 240;
    const startX = VIEW_W / 2 - ((columns - 1) * spacing) / 2;

    result.set(node.id, {
      x: startX + col * spacing,
      y: VIEW_H - 75 - row * 75,
    });
  });

  return result;
}

export default function StoryGraph({
  nodes,
  edges,
  selectedId,
  onSelect,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const { t, tx, lang } = useLang();
  const arabic = lang === "ar";
  const svgRef = useRef<SVGSVGElement | null>(null);

  const degrees = useMemo(() => {
    const map = new Map(nodes.map((n) => [n.id, 0]));
    for (const edge of edges) {
      map.set(edge.from, (map.get(edge.from) ?? 0) + 1);
      map.set(edge.to, (map.get(edge.to) ?? 0) + 1);
    }
    return map;
  }, [nodes, edges]);

  const maxDegree = Math.max(0, ...degrees.values());

  const initialLayout = useMemo(
    () => buildInitialLayout(nodes, edges),
    [nodes, edges]
  );

  const [positions, setPositions] = useState<Map<string, Point>>(
    () => new Map(initialLayout)
  );

  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [resetHot, setResetHot] = useState(false);

  const dragRef = useRef<{
    id: string;
    dx: number;
    dy: number;
    moved: boolean;
  } | null>(null);

  const suppressClick = useRef(false);

  const graphKey = useMemo(
    () =>
      [
        nodes.map((n) => n.id).sort().join(","),
        edges
          .map((e) => `${e.from}>${e.to}:${JSON.stringify(e.label)}`)
          .sort()
          .join(","),
      ].join("|"),
    [nodes, edges]
  );

  useEffect(() => {
    setPositions(new Map(initialLayout));
  }, [graphKey, initialLayout]);

  const sizes = useMemo(() => {
    const map = new Map<string, Size>();

    for (const node of nodes) {
      const important = (degrees.get(node.id) ?? 0) === maxDegree && maxDegree > 0;
      map.set(node.id, {
        w: nodeWidth(tx(node.label), node.type === "place", arabic, important),
        h: node.id === selectedId ? SELECTED_H : NODE_H,
      });
    }

    return map;
  }, [nodes, degrees, maxDegree, tx, arabic, selectedId]);

  const routedEdges = useMemo<RoutedEdge[]>(() => {
    const groups = new Map<string, { edge: GraphEdge; index: number }[]>();

    edges.forEach((edge, index) => {
      const key = pairKey(edge.from, edge.to);
      const group = groups.get(key) ?? [];
      group.push({ edge, index });
      groups.set(key, group);
    });

    const routed: RoutedEdge[] = [];

    for (const group of groups.values()) {
      group.forEach(({ edge, index }, parallelIndex) => {
        routed.push({
          edge,
          index,
          parallelIndex,
          parallelCount: group.length,
        });
      });
    }

    return routed.sort((a, b) => a.index - b.index);
  }, [edges]);

  function clamp(id: string, point: Point): Point {
    const size = sizes.get(id) ?? { w: 140, h: NODE_H };

    return {
      x: Math.max(
        size.w / 2 + BOUNDS_PAD,
        Math.min(VIEW_W - size.w / 2 - BOUNDS_PAD, point.x)
      ),
      y: Math.max(
        size.h / 2 + BOUNDS_PAD,
        Math.min(VIEW_H - size.h / 2 - BOUNDS_PAD, point.y)
      ),
    };
  }

  function clientToView(clientX: number, clientY: number): Point {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };

    return {
      x: ((clientX - rect.left) / rect.width) * VIEW_W,
      y: ((clientY - rect.top) / rect.height) * VIEW_H,
    };
  }

  function onNodePointerDown(e: React.PointerEvent<SVGGElement>, id: string) {
    if (e.button !== 0) return;

    const current = positions.get(id);
    if (!current) return;

    const point = clientToView(e.clientX, e.clientY);

    suppressClick.current = false;
    dragRef.current = {
      id,
      dx: point.x - current.x,
      dy: point.y - current.y,
      moved: false,
    };

    setDraggingId(id);
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  function onNodePointerMove(e: React.PointerEvent<SVGGElement>) {
    const drag = dragRef.current;
    if (!drag) return;

    const point = clientToView(e.clientX, e.clientY);

    setPositions((previous) => {
      const old = previous.get(drag.id);
      if (!old) return previous;

      const next = clamp(drag.id, {
        x: point.x - drag.dx,
        y: point.y - drag.dy,
      });

      if (Math.hypot(next.x - old.x, next.y - old.y) > DRAG_SLOP) {
        drag.moved = true;
        suppressClick.current = true;
      }

      const updated = new Map(previous);
      updated.set(drag.id, next);
      return updated;
    });
  }

  function onNodePointerUp(e: React.PointerEvent<SVGGElement>) {
    dragRef.current = null;
    setDraggingId(null);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
  }

  function resetLayout() {
    setPositions(new Map(initialLayout));
  }

  function nudge(id: string, key: string) {
    const delta: Record<string, Point> = {
      ArrowUp: { x: 0, y: -16 },
      ArrowDown: { x: 0, y: 16 },
      ArrowLeft: { x: -16, y: 0 },
      ArrowRight: { x: 16, y: 0 },
    };

    const d = delta[key];
    if (!d) return;

    setPositions((previous) => {
      const current = previous.get(id);
      if (!current) return previous;

      const updated = new Map(previous);
      updated.set(
        id,
        clamp(id, {
          x: current.x + d.x,
          y: current.y + d.y,
        })
      );
      return updated;
    });
  }

  const posOf = (id: string): Point =>
    positions.get(id) ?? initialLayout.get(id) ?? { x: VIEW_W / 2, y: VIEW_H / 2 };

  return (
    <div className="overflow-hidden rounded-2xl border border-stone-200 bg-stone-50 shadow-sm">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        className="block min-h-[620px] w-full touch-none select-none"
        role="group"
        aria-label="Story map"
      >
        <defs>
          <marker
            id="story-map-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#78716C" />
          </marker>

          <filter id="node-shadow" x="-30%" y="-40%" width="160%" height="180%">
            <feDropShadow dx="0" dy="3" stdDeviation="5" floodOpacity="0.10" />
          </filter>
        </defs>

        {routedEdges.map(({ edge, index, parallelIndex, parallelCount }) => {
          const source = posOf(edge.from);
          const target = posOf(edge.to);

          const targetSize = sizes.get(edge.to) ?? { w: 140, h: NODE_H };
          const sourceSize = sizes.get(edge.from) ?? { w: 140, h: NODE_H };

          const start = boundaryPoint(target, source, sourceSize);
          const end = boundaryPoint(source, target, targetSize);

          const dx = end.x - start.x;
          const dy = end.y - start.y;
          const distance = Math.hypot(dx, dy) || 1;

          const nx = -dy / distance;
          const ny = dx / distance;

          const centeredIndex = parallelIndex - (parallelCount - 1) / 2;

          // Give parallel relations enough separation that both the paths
          // and their labels remain readable in the default layout.
          const curvature =
            parallelCount === 1 ? 0 : centeredIndex * 120;

          const control = {
            x: (start.x + end.x) / 2 + nx * curvature,
            y: (start.y + end.y) / 2 + ny * curvature,
          };

          const baseLabelPoint = {
            x: 0.25 * start.x + 0.5 * control.x + 0.25 * end.x,
            y: 0.25 * start.y + 0.5 * control.y + 0.25 * end.y,
          };

          // Move labels farther apart than the curves themselves. This is
          // especially important for two relations sharing the same nodes.
          const labelSpread =
            parallelCount === 1 ? 0 : centeredIndex * 34;

          const labelPoint = {
            x: baseLabelPoint.x + nx * labelSpread,
            y: baseLabelPoint.y + ny * labelSpread,
          };

          const path = `M ${start.x} ${start.y} Q ${control.x} ${control.y} ${end.x} ${end.y}`;

          const label = tx(edge.label);
          const labelW = Math.max(
            72,
            label.length * (arabic ? 9.2 : 7.3) + 24
          );

          const symmetric = isSymmetric(label);

          return (
            <g
              key={`${edge.from}-${edge.to}-${index}-${label}`}
              className="transition-opacity"
            >
              <path
                d={path}
                fill="none"
                stroke="#A8A29E"
                strokeWidth={2.2}
                strokeLinecap="round"
                markerEnd={symmetric ? undefined : "url(#story-map-arrow)"}
              />

              <rect
                x={labelPoint.x - labelW / 2}
                y={labelPoint.y - 14}
                width={labelW}
                height={28}
                rx={14}
                fill="#FFFFFF"
                stroke="#E7E5E4"
              />

              <text
                x={labelPoint.x}
                y={labelPoint.y}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={14}
                fontWeight={500}
                fill="#57534E"
              >
                {label}
              </text>
            </g>
          );
        })}

        {nodes.map((node) => {
          const p = posOf(node.id);
          const selected = node.id === selectedId;
          const label = tx(node.label);
          const isPlace = node.type === "place";
          const degree = degrees.get(node.id) ?? 0;
          const important = degree === maxDegree && maxDegree > 0;
          const isolated = degree === 0;
          const { w, h } = sizes.get(node.id) ?? { w: 140, h: NODE_H };

          return (
            <g
              key={node.id}
              data-node={node.id}
              role="button"
              tabIndex={0}
              aria-pressed={selected}
              opacity={isolated ? 0.72 : 1}
              filter={important || selected ? "url(#node-shadow)" : undefined}
              className={`focus:outline-none ${
                draggingId === node.id ? "cursor-grabbing" : "cursor-grab"
              }`}
              onPointerDown={(e) => onNodePointerDown(e, node.id)}
              onPointerMove={onNodePointerMove}
              onPointerUp={onNodePointerUp}
              onPointerCancel={onNodePointerUp}
              onClick={() => {
                if (suppressClick.current) {
                  suppressClick.current = false;
                  return;
                }
                onSelect(node.id);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(node.id);
                } else if (e.key.startsWith("Arrow")) {
                  e.preventDefault();
                  nudge(node.id, e.key);
                }
              }}
            >
              {important && !selected && (
                <rect
                  x={p.x - w / 2 - 7}
                  y={p.y - h / 2 - 7}
                  width={w + 14}
                  height={h + 14}
                  rx={(h + 14) / 2}
                  fill="#7A003C"
                  fillOpacity={0.07}
                />
              )}

              {selected && (
                <rect
                  x={p.x - w / 2 - 6}
                  y={p.y - h / 2 - 6}
                  width={w + 12}
                  height={h + 12}
                  rx={(h + 12) / 2}
                  fill="none"
                  stroke="#7A003C"
                  strokeOpacity={0.28}
                  strokeWidth={4}
                />
              )}

              <rect
                x={p.x - w / 2}
                y={p.y - h / 2}
                width={w}
                height={h}
                rx={18}
                fill={
                  selected
                    ? "#63002F"
                    : important
                      ? "#FFF7FA"
                      : isPlace
                        ? "#FAFAF9"
                        : "#FFFFFF"
                }
                stroke={
                  selected || important
                    ? "#7A003C"
                    : isPlace
                      ? "#D6D3D1"
                      : "#E7E5E4"
                }
                strokeWidth={important ? 2.2 : 1.5}
              />

              {isPlace && (
                <g
                  transform={`translate(${p.x - w / 2 + 24}, ${p.y}) scale(0.7)`}
                  fill="none"
                  stroke={selected ? "#FFFFFF" : "#7A003C"}
                  strokeWidth={2.4}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M0 11s7-5.5 7-11a7 7 0 1 0-14 0c0 5.5 7 11 7 11Z"
                    transform="translate(0,-12)"
                  />
                  <circle cx="0" cy="-14" r="2.5" />
                </g>
              )}

              <text
                x={isPlace ? p.x + 10 : p.x}
                y={p.y}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={important || selected ? 17 : 15}
                fontWeight={important || selected ? 650 : 500}
                fill={selected ? "#FFFFFF" : "#1C1917"}
              >
                {label}
              </text>
            </g>
          );
        })}

        <g
          data-graph-control="reset"
          role="button"
          tabIndex={0}
          aria-label={t("resetLayout")}
          className="cursor-pointer focus:outline-none"
          onClick={resetLayout}
          onPointerEnter={() => setResetHot(true)}
          onPointerLeave={() => setResetHot(false)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              resetLayout();
            }
          }}
        >
          <title>{t("resetLayout")}</title>
          <circle
            cx={VIEW_W - 42}
            cy={42}
            r={20}
            fill="#FFFFFF"
            stroke={resetHot ? "#7A003C" : "#D6D3D1"}
            strokeWidth={1.5}
          />
          <g
            transform={`translate(${VIEW_W - 42}, 42) scale(0.82) translate(-12, -12)`}
            fill="none"
            stroke={resetHot ? "#7A003C" : "#78716C"}
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 12a9 9 0 1 0 3-6.7L3 8m0 0V3m0 5h5"
            />
          </g>
        </g>
      </svg>
    </div>
  );
}
