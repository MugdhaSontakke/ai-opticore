# Performance

This section describes how AI-OptiCore measures performance. We never publish
universal reduction/latency claims because real numbers depend on your
workload, model, and hardware. Run your own benchmarks.

## Benchmark methodology

The benchmark (`ai-opticore benchmark`, or `BenchmarkRunner` in Python)
measures, for a set of sample requests with `repeats`:

**Baseline (no optimization):**

- Input tokens as counted by the real tokenizer before any transformation.
- Output tokens reported by the provider.
- Latency (wall time) of the provider call.

**Optimized (AI-OptiCore enabled):**

- Input tokens after the full pipeline.
- Output tokens reported by the provider.
- Latency of the (optimized) provider call.

**Derived metrics:**

- Token reduction `%` = `(baseline_in - optimized_in) / baseline_in * 100`.
- Latency change `%` = `(optimized_lat - baseline_lat) / baseline_lat * 100`.
- Cache hit rate (when a semantic cache is in the loop).
- Mean values across samples/repeats are reported; individual samples are in
  the metrics summary.

If a metric cannot be measured it is rendered as `N/A`, never invented.

## Metrics

The `MetricsCollector` tracks counters (`request_count`, `cache_hits`,
`cache_misses`, `original_tokens`, `optimized_tokens`, `tokens_saved`,
`errors`, `model_usage`) and value series (`latency_ms`, `reduction_percent`,
...). Series expose min/max/mean and support custom aggregators.

## Quality evaluation

Token reduction alone is not enough. AI-OptiCore compares original vs
optimized requests and responses:

- Request similarity (character/token-level by default; plug in
  embedding-based evaluators).
- Response similarity (same).
- `QualityGate` warns when similarity falls below a configurable threshold and
  when aggressive safety mode is enabled.

We never claim "X% reduction with no quality loss" unless an evaluation in
your environment demonstrates it.

## Reproducibility

To reproduce a benchmark:

1. Record the exact `OptimizationConfig` used (included in `BenchmarkResult.config`).
2. Record the model, provider, and detected hardware (all included in the result).
3. Use the same sample set and repeats; share the sample JSON with the results.
4. Note that latency and throughput are environment-dependent (network, load,
   GPU contention).

### Memory

`MemoryTracker` uses OS-level `ru_maxrss`. Reports peak RSS in bytes; on
platforms where it is unavailable it reports `0` and notes the limitation.

## Benchmark checklist

- [ ] Config recorded
- [ ] Model/provider recorded
- [ ] Hardware recorded
- [ ] Samples reproducible
- [ ] N/A used for unmeasurable metrics
- [ ] Quality evaluation run when claiming quality preservation