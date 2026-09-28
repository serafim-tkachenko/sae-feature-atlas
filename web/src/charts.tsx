import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent } from "react";
import type { Feature } from "./types";

export const numeric = (x: unknown): x is number =>
  typeof x === "number" && Number.isFinite(x);
export const number = (x: number | null | undefined) =>
  numeric(x)
    ? x.toLocaleString(undefined, { maximumFractionDigits: 3 })
    : "Not recorded";
export const percent = (x: number | null) =>
  numeric(x)
    ? `${(x * 100).toLocaleString(undefined, { maximumFractionDigits: 3 })}%`
    : "Not recorded";
const tick = (x: number) =>
  x >= 1000 ? `${+(x / 1000).toFixed(1)}k` : `${+x.toPrecision(2)}`;

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(240, entry.contentRect.width)),
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}

export function Scatter({
  features,
  selected,
  brushed,
  onSelect,
  onBrush,
}: {
  features: Feature[];
  selected: number | null;
  brushed: number[] | null;
  onSelect: (id: number) => void;
  onBrush: (ids: number[] | null) => void;
}) {
  const { ref, width } = useWidth();
  const [drag, setDrag] = useState<{
    start: [number, number];
    end: [number, number];
  } | null>(null);
  const [hover, setHover] = useState<Feature | null>(null);
  const brushIds = useMemo(
    () => (brushed === null ? null : new Set(brushed)),
    [brushed],
  );
  const data = features.filter(
    (f) =>
      numeric(f.frequency) && numeric(f.p99) && f.frequency >= 0 && f.p99 >= 0,
  );
  const height = 225,
    left = 58,
    right = width - 16,
    top = 14,
    bottom = 177;
  const xmax =
    data.reduce((max, f) => Math.max(max, f.frequency!), 0.00001) * 1.05;
  const ymax =
    data.reduce((max, f) => Math.max(max, Math.log1p(f.p99!)), 1) * 1.05;
  const x = (f: Feature) => left + (f.frequency! / xmax) * (right - left);
  const y = (f: Feature) =>
    bottom - (Math.log1p(f.p99!) / ymax) * (bottom - top);
  function coords(event: PointerEvent<SVGSVGElement>): [number, number] {
    const box = event.currentTarget.getBoundingClientRect();
    return [
      Math.max(
        left,
        Math.min(right, ((event.clientX - box.left) / box.width) * width),
      ),
      Math.max(
        top,
        Math.min(bottom, ((event.clientY - box.top) / box.height) * height),
      ),
    ];
  }
  function nearest(point: [number, number]) {
    let match: Feature | null = null,
      distance = 20;
    for (const f of data) {
      const d = Math.hypot(x(f) - point[0], y(f) - point[1]);
      if (d < distance) {
        match = f;
        distance = d;
      }
    }
    return match;
  }
  return (
    <div className="scatter" ref={ref}>
      <div className="section-heading">
        <div>
          <h2>Feature landscape</h2>
          <p>Frequency × activation strength</p>
        </div>
        {brushed && (
          <button onClick={() => onBrush(null)}>Clear chart selection</button>
        )}
      </div>
      {data.length ? (
        <>
          <svg
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label="Feature frequency versus p99 activation. Use the feature list and numeric filters for keyboard access."
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              const point = coords(event);
              setDrag({ start: point, end: point });
            }}
            onPointerMove={(event) => {
              const point = coords(event);
              if (drag) setDrag({ ...drag, end: point });
              else setHover(nearest(point));
            }}
            onPointerLeave={() => setHover(null)}
            onPointerCancel={() => setDrag(null)}
            onPointerUp={(event) => {
              if (!drag) return;
              const end = coords(event);
              if (
                Math.hypot(end[0] - drag.start[0], end[1] - drag.start[1]) < 6
              ) {
                const match = nearest(end);
                if (match) onSelect(match.id);
              } else {
                const [x0, x1] = [drag.start[0], end[0]].sort((a, b) => a - b);
                const [y0, y1] = [drag.start[1], end[1]].sort((a, b) => a - b);
                onBrush(
                  data
                    .filter(
                      (f) =>
                        x(f) >= x0 && x(f) <= x1 && y(f) >= y0 && y(f) <= y1,
                    )
                    .map((f) => f.id),
                );
              }
              setDrag(null);
            }}
          >
            {[0, 1, 2, 3].map((i) => (
              <g key={i}>
                <line
                  x1={left}
                  x2={right}
                  y1={bottom - (i / 3) * (bottom - top)}
                  y2={bottom - (i / 3) * (bottom - top)}
                  className="gridline"
                />
                <text
                  x={left - 8}
                  y={bottom - (i / 3) * (bottom - top) + 4}
                  textAnchor="end"
                >
                  {tick(Math.expm1((i / 3) * ymax))}
                </text>
                <text
                  x={left + (i / 3) * (right - left)}
                  y={bottom + 19}
                  textAnchor={i === 3 ? "end" : "middle"}
                >
                  {+((i / 3) * xmax * 100).toPrecision(2)}%
                </text>
              </g>
            ))}
            <text
              transform={`translate(13 ${(top + bottom) / 2}) rotate(-90)`}
              textAnchor="middle"
            >
              p99 activation (log scale)
            </text>
            <text x={(left + right) / 2} y={height - 4} textAnchor="middle">
              Recorded token frequency
            </text>
            {data.map((f) => (
              <circle
                key={f.id}
                cx={x(f)}
                cy={y(f)}
                r={f.id === selected ? 5 : 2.6}
                className={f.id === selected ? "point selected" : "point"}
                opacity={brushIds && !brushIds.has(f.id) ? 0.1 : 0.65}
              >
                <title>
                  Feature {f.id}: {percent(f.frequency)}, p99 {number(f.p99)}
                </title>
              </circle>
            ))}
            {drag && (
              <rect
                x={Math.min(drag.start[0], drag.end[0])}
                y={Math.min(drag.start[1], drag.end[1])}
                width={Math.abs(drag.end[0] - drag.start[0])}
                height={Math.abs(drag.end[1] - drag.start[1])}
                className="brush"
              />
            )}
          </svg>
          <p className="chart-caption">
            {hover
              ? `Feature ${hover.id} · ${percent(hover.frequency)} · p99 ${number(hover.p99)}`
              : "Click a point to inspect. Drag a region to filter the list."}
            {data.length !== features.length &&
              ` ${features.length - data.length} features lack chart values.`}
          </p>
        </>
      ) : (
        <div className="empty small">
          No features with both frequency and p99 activation in this view.
        </div>
      )}
    </div>
  );
}

export function Histogram({ feature }: { feature: Feature }) {
  const { ref, width } = useWidth();
  const h = feature.histogram;
  const height = 160,
    left = 50,
    right = width - 18,
    bottom = 116,
    top = 12;
  const maximum = h ? Math.max(1, ...h.counts) : 1;
  return (
    <div className="histogram" ref={ref}>
      <div className="section-heading">
        <h3>Stored activation distribution</h3>
        <span>
          {h ? `${number(h.count)} finite stored rows` : "Unavailable"}
        </span>
      </div>
      {h ? (
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Histogram of ${h.count} stored activation values for feature ${feature.id}, before analysis filtering.`}
        >
          {[0, 0.5, 1].map((part) => (
            <g key={part}>
              <line
                x1={left}
                x2={right}
                y1={bottom - part * (bottom - top)}
                y2={bottom - part * (bottom - top)}
                className="gridline"
              />
              <text
                x={left - 6}
                y={bottom - part * (bottom - top) + 4}
                textAnchor="end"
              >
                {tick(part * maximum)}
              </text>
            </g>
          ))}
          {h.counts.map((count, i) => (
            <rect
              key={i}
              x={left + (i * (right - left)) / h.counts.length}
              y={bottom - (count / maximum) * (bottom - top)}
              width={Math.max(1, (right - left) / h.counts.length - 1)}
              height={(count / maximum) * (bottom - top)}
              className="hist-bar"
            >
              <title>
                {number(h.edges[i])}–{number(h.edges[i + 1])}: {number(count)}{" "}
                rows
              </title>
            </rect>
          ))}
          {[0, Math.floor(h.counts.length / 2), h.counts.length].map(
            (index, i) => (
              <text
                key={index}
                x={left + (index / h.counts.length) * (right - left)}
                y={bottom + 18}
                textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
              >
                {tick(h.edges[index])}
              </text>
            ),
          )}
          <text x={(left + right) / 2} y={height - 2} textAnchor="middle">
            Activation magnitude
          </text>
          <text
            transform={`translate(12 ${(top + bottom) / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            Rows
          </text>
        </svg>
      ) : (
        <p className="muted">
          The sparse activation artifact is missing, empty, unreadable, or its
          storage mode is ambiguous.
        </p>
      )}
      <p className="fine">
        Stored rows before analysis filtering. Under top-k storage, only
        retained activations are represented; omitted activations are not zeros.
      </p>
    </div>
  );
}
