import type { BenchmarkData } from "../App";

function Metric({ label, value }: { label: string; value: string | number | null }) {
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

export default function BenchmarkView({ data }: { data: BenchmarkData }) {
  return (
    <div>
      <Section title="Configuration">
        <Metric label="Model" value={data.model} />
        <Metric label="Provider" value={data.provider} />
        <Metric label="Hardware" value={data.hardware} />
        <Metric label="Samples" value={data.samples} />
      </Section>

      <Section title="Baseline (no optimization)">
        <Metric label="Input tokens" value={data.baseline.input_tokens} />
        <Metric label="Output tokens" value={data.baseline.output_tokens} />
        <Metric label="Latency" value={data.baseline.latency_ms != null ? `${data.baseline.latency_ms} ms` : null} />
      </Section>

      <Section title="Optimized (AI-OptiCore enabled)">
        <Metric label="Input tokens" value={data.optimized.input_tokens} />
        <Metric label="Output tokens" value={data.optimized.output_tokens} />
        <Metric label="Latency" value={data.optimized.latency_ms != null ? `${data.optimized.latency_ms} ms` : null} />
      </Section>

      <Section title="Comparison">
        <Metric label="Token reduction" value={data.metrics.token_reduction_percent} />
        <Metric label="Latency change" value={data.metrics.latency_change_percent} />
        <Metric label="Cache hit rate" value={data.metrics.cache_hit_rate} />
        <Metric label="Avg tokens saved / request" value={data.metrics.avg_tokens_saved_per_request} />
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