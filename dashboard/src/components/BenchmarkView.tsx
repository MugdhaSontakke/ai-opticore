import type { BenchmarkData } from "../App";

function Metric({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div style={{ marginBottom: "0.5rem" }}>
      <strong>{label}: </strong>
      <span style={{ fontVariantNumeric: "tabular-nums" }}>
        {value === null || value === undefined ? "N/A" : String(value)}
      </span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section
      style={{
        marginBottom: "1.5rem",
        padding: "1rem",
        border: "1px solid #ddd",
        borderRadius: 8,
      }}
    >
      <h2 style={{ marginTop: 0, fontSize: "1rem" }}>{title}</h2>
      {children}
    </section>
  );
}

function latency(
  value: number | null | undefined,
  median: number | null | undefined,
  p95: number | null | undefined,
) {
  if (value === null || value === undefined) return null;
  const parts = [`avg ${value} ms`];
  if (typeof median === "number") parts.push(`median ${median} ms`);
  if (typeof p95 === "number") parts.push(`p95 ${p95} ms`);
  return parts.join(" · ");
}

export default function BenchmarkView({ data }: { data: BenchmarkData }) {
  const env = data.environment;
  return (
    <div>
      <Section title="Configuration">
        <Metric label="Model" value={data.model} />
        <Metric label="Provider" value={data.provider} />
        <Metric label="Hardware" value={data.hardware} />
        <Metric label="Samples" value={data.samples} />
        <Metric label="Dataset" value={data.dataset ?? "builtin"} />
        {env && (
          <>
            <Metric label="Timestamp" value={env.timestamp ?? null} />
            <Metric label="Version" value={env.opticore_version ?? null} />
            <Metric label="Python" value={env.python_version ?? null} />
            <Metric label="OS" value={env.os ?? null} />
          </>
        )}
      </Section>

      <Section title="Baseline (no optimization)">
        <Metric label="Input tokens" value={data.baseline.input_tokens} />
        <Metric label="Output tokens" value={data.baseline.output_tokens} />
        <Metric
          label="Latency"
          value={latency(
            data.baseline.latency_ms,
            data.baseline.latency_median_ms,
            data.baseline.latency_p95_ms,
          )}
        />
      </Section>

      <Section title="Optimized (AI-OptiCore enabled)">
        <Metric label="Input tokens" value={data.optimized.input_tokens} />
        <Metric label="Output tokens" value={data.optimized.output_tokens} />
        <Metric
          label="Latency"
          value={latency(
            data.optimized.latency_ms,
            data.optimized.latency_median_ms,
            data.optimized.latency_p95_ms,
          )}
        />
      </Section>

      <Section title="Comparison">
        <Metric label="Token reduction" value={data.metrics.token_reduction_percent} />
        <Metric label="Latency change" value={data.metrics.latency_change_percent} />
        <Metric label="Cache hit rate" value={data.metrics.cache_hit_rate} />
        <Metric label="Avg tokens saved / request" value={data.metrics.avg_tokens_saved_per_request} />
        <Metric
          label="Optimizer overhead (avg)"
          value={data.metrics.optimizer_overhead_ms_avg ?? null}
        />
        <Metric
          label="Provider latency (avg)"
          value={data.metrics.provider_latency_ms_avg ?? null}
        />
        <Metric label="Quality score (avg)" value={data.metrics.quality_score_avg ?? null} />
      </Section>

      <Section title="Estimated cost">
        <Metric label="Baseline" value={data.metrics.estimated_cost_baseline ?? null} />
        <Metric label="Optimized" value={data.metrics.estimated_cost_optimized ?? null} />
        <Metric label="Savings" value={data.metrics.estimated_cost_savings ?? null} />
      </Section>

      {data.notes.length > 0 && (
        <Section title="Notes">
          <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>
            {data.notes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  );
}