# ROCm / AMD Hardware Support

> Status: **experimental**. AI-OptiCore provides the abstraction and honest
> detection for AMD acceleration, but does not claim every AMD GPU is
> supported. This document records requirements, configuration, tested
> combinations, limitations, and the benchmarking procedure.

## Goals

- Allow AMD acceleration to be added and tested **independently** of the core.
- The project must **not** depend on AMD hardware to import, test, or run
  (CPU-only environments work; ROCm detection degrades gracefully).
- Investigate compatibility with PyTorch, ROCm, and ONNX Runtime where
  supported — and document what is actually verified.

## Requirements

- Linux host with an AMD GPU (GFX9+ typically; see your vendor matrix).
- ROCm runtime installed system-wide (e.g. via `amdgpu-install` or ROCm
  docker images `rocm/pytorch`).
- A ROCm-enabled PyTorch build:
  `pip install torch --index-url https://download.pytorch.org/whl/rocm<version>/`
- `rocm-smi` for health checks (optional but useful).

## Installation

```bash
# Install AI-OptiCore with the inference extra
pip install -e ".[huggingface,benchmark]"

# Verify PyTorch sees the ROCm-enabled build
python -c "import torch; print(torch.version.hip)"
```

Confirm detection through AI-OptiCore:

```bash
ai-opticore hardware
```

If `rocm` is listed as `available`, the environment detected a ROCm PyTorch
build with a visible AMD device.

## Supported configurations

The detection logic requires **both**:

1. `torch.version.hip` is set (i.e. a ROCm build of PyTorch), and
2. `torch.cuda.device_count() > 0` with a device name containing `amd`,
   `radeon`, or `instinct`.

The table below is a template. Fill actual tested combinations during hardware
validation — do not mark combinations as verified without hardware.

| OS | GPU | ROCm | PyTorch | Verified |
| --- | --- | --- | --- | --- |
| (untested) | — | — | — | No — needs hardware |

> No real AMD performance results are reported in this repository. Until a
> maintainer or contributor validates against real AMD hardware, the backend
> remains **experimental and unverified** for performance claims.

## Hardware detection contract

`ROCmBackend.available()` returns `True` **only** when an AMD device was
actually detected. Otherwise it returns `False` and `capabilities()`/
`details()` explain why. This is intentional: we never fake hardware info.

## Known limitations

- Only text-based detection; no automatic kernel-level tuning.
- No ONNX Runtime execution-provider integration yet (planned).
- Computational kernels depend entirely on the PyTorch ROCm build, not on
  AI-OptiCore. Model-level support (e.g. FlashAttention on ROCm) is the
  concern of the framework used.
- No CI environment currently exercises real ROCm hardware.

## Benchmarking procedure (for hardware owners)

1. Make sure `ai-opticore hardware` shows `rocm: available`.
2. Run a conservative set of samples through the local provider:

```bash
ai-opticore benchmark --provider huggingface --model <hf-model-id>
```

3. Record: sample set, config, model, provider, hardware details, wall-time,
   token counts, and peak memory (`MemoryTracker`).
4. Compare against the same workload on CPU on the same machine:

```bash
ai-opticore benchmark --provider huggingface --model <hf-model-id>
```

(a CPU vs ROCm delta is only meaningful when both runs use identical models,
tokenizers, batch sizes, and sample sets.)

5. Publish results via a **Benchmark Report** issue/PR; never commit
   fabricated numbers.

## Layout

- `src/opticore/hardware/base.py` — `BaseHardwareBackend`, `ROCmBackend`.
- `src/opticore/hardware/__init__.py` — detection and capability reporting.
- `tests/test_routing_hardware.py` — detection tests (mock-friendly).