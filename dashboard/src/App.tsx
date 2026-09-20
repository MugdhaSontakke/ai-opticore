import { useState, useRef } from "react";
import BenchmarkView from "./components/BenchmarkView";

export interface BenchmarkData {
  model: string;
  provider: string;
  hardware: string;
  samples: number;
  dataset?: string;
  environment?: {
    timestamp?: string;
    opticore_version?: string;
    python_version?: string;
    os?: string;
  };
  baseline: {
    input_tokens: number | null;
    output_tokens: number | null;
    latency_ms: number | null;
    latency_median_ms?: number | null;
    latency_p95_ms?: number | null;
  };
  optimized: {
    input_tokens: number | null;
    output_tokens: number | null;
    latency_ms: number | null;
    latency_median_ms?: number | null;
    latency_p95_ms?: number | null;
  };
  metrics: {
    token_reduction_percent: string | number;
    latency_change_percent: string | number;
    cache_hit_rate: string | number;
    avg_tokens_saved_per_request: number;
    optimizer_overhead_ms_avg?: number | null;
    provider_latency_ms_avg?: number | null;
    quality_score_avg?: string | number;
    estimated_cost_baseline?: string | number;
    estimated_cost_optimized?: string | number;
    estimated_cost_savings?: string | number;
    scenarios?: Record<string, unknown>;
    summary: Record<string, unknown>;
  };
  notes: string[];
}

// DEMO DATA — explicitly labeled. It only demonstrates the file shape and is
// NEVER presented as a measured result. Improvement claims are N/A.
const DEMO_DATA: BenchmarkData = {
  model: "example-model",
  provider: "example-provider",
  hardware: "example-backend",
  samples: 0,
  baseline: { input_tokens: null, output_tokens: null, latency_ms: null },
  optimized: { input_tokens: null, output_tokens: null, latency_ms: null },
  metrics: {
    token_reduction_percent: "N/A",
    latency_change_percent: "N/A",
    cache_hit_rate: "N/A",
    avg_tokens_saved_per_request: 0,
    summary: { counters: {} },
  },
  notes: [
    "DEMO / SAMPLE DATA — no benchmark was run. Load a real ai-opticore benchmark --json output file to see measured results.",
  ],
};

type DataKind = "none" | "demo" | "real";

export default function App() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [kind, setKind] = useState<DataKind>("none");
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setData(JSON.parse(reader.result as string));
        setKind("real");
        setError(null);
      } catch {
        setError("Invalid JSON file");
      }
    };
    reader.readAsText(file);
  }

  function loadDemo() {
    setData(DEMO_DATA);
    setKind("demo");
    setError(null);
  }

  const badge =
    kind === "real" ? (
      <span style={{ color: "green", fontWeight: 600 }}>REAL MEASURED DATA</span>
    ) : kind === "demo" ? (
      <span style={{ color: "darkorange", fontWeight: 600 }}>DEMO / SAMPLE DATA</span>
    ) : (
      <span style={{ color: "#888", fontWeight: 600 }}>DATA UNAVAILABLE</span>
    );

  return (
    <div style={{ padding: "2rem", fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <header style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ margin: 0 }}>AI-OptiCore Dashboard</h1>
        <p style={{ color: "#666", margin: "0.5rem 0 0 0" }}>
          Benchmark results viewer — load a real <code>--json</code> benchmark output file
        </p>
        {badge && <div style={{ marginTop: "0.75rem" }}>{badge}</div>}
      </header>

      <div style={{ marginBottom: "1.5rem" }}>
        <input ref={inputRef} type="file" accept=".json" onChange={handleFile} />
        <button style={{ marginLeft: "0.5rem", padding: "0.4rem 0.8rem" }} onClick={loadDemo}>
          Load demo data
        </button>
      </div>

      {error && <div style={{ color: "crimson", marginBottom: "1rem" }}>{error}</div>}

      {data ? (
        <BenchmarkView data={data} />
      ) : (
        <div
          style={{
            padding: "1rem",
            border: "1px dashed #ccc",
            borderRadius: 8,
            color: "#777",
          }}
        >
          <p style={{ margin: 0 }}>
            No benchmark data loaded. Status: <strong>UNAVAILABLE</strong> —
            upload a real <code>ai-opticore benchmark --json</code> output file
            or load the demo data to preview the layout.
          </p>
        </div>
      )}
    </div>
  );
}