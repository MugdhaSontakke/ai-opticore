import { useState, useRef } from "react";
import BenchmarkView from "./components/BenchmarkView";

export interface BenchmarkData {
  model: string;
  provider: string;
  hardware: string;
  samples: number;
  baseline: {
    input_tokens: number | null;
    output_tokens: number | null;
    latency_ms: number | null;
  };
  optimized: {
    input_tokens: number | null;
    output_tokens: number | null;
    latency_ms: number | null;
  };
  metrics: {
    token_reduction_percent: string | number;
    latency_change_percent: string | number;
    cache_hit_rate: string | number;
    avg_tokens_saved_per_request: number;
    summary: Record<string, unknown>;
  };
  notes: string[];
}

const SAMPLE_DATA: BenchmarkData = {
  model: "gpt-4o-mini",
  provider: "openai",
  hardware: "cpu",
  samples: 3,
  baseline: { input_tokens: 42, output_tokens: 11, latency_ms: 823.4 },
  optimized: { input_tokens: 38, output_tokens: 11, latency_ms: 791.2 },
  metrics: {
    token_reduction_percent: "9.52%",
    latency_change_percent: "-3.91%",
    cache_hit_rate: "N/A",
    avg_tokens_saved_per_request: 4,
    summary: { counters: { request_count: 3 } },
  },
  notes: [],
};

export default function App() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setData(JSON.parse(reader.result as string));
        setError(null);
      } catch {
        setError("Invalid JSON file");
      }
    };
    reader.readAsText(file);
  }

  return (
    <div style={{ padding: "2rem", fontFamily: "system-ui, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <header style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ margin: 0 }}>AI-OptiCore Dashboard</h1>
        <p style={{ color: "#666", margin: "0.5rem 0 0 0" }}>
          Benchmark results viewer — load a <code>--json</code> benchmark output file
        </p>
      </header>

      <div style={{ marginBottom: "1.5rem" }}>
        <input ref={inputRef} type="file" accept=".json" onChange={handleFile} />
        <button
          style={{ marginLeft: "0.5rem", padding: "0.4rem 0.8rem" }}
          onClick={() => setData(SAMPLE_DATA)}
        >
          Load sample data
        </button>
      </div>

      {error && <div style={{ color: "crimson", marginBottom: "1rem" }}>{error}</div>}

      {data ? <BenchmarkView data={data} /> : <p>No data loaded.</p>}
    </div>
  );
}