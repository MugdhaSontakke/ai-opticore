# Getting Started

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/ai-opticore/ai-opticore.git
cd ai-opticore
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

Optional extras: `openai`, `ollama`, `huggingface`, `semantic`, `benchmark`,
`dev`.

## Quick start

```python
from opticore import Optimizer

optimizer = Optimizer()

result = optimizer.optimize(
    prompt="Can   you   please explain   what a  token  is?",
    system="You are a helpful assistant.",
)

print(result.optimized_prompt)
print(result.tokens_saved)
print(result.reduction_percent)
```

## First optimization

Let's walk through the full flow with the bundled fake provider (no API key
needed), then swap in a real one.

```python
from opticore import AIClient
from opticore.providers.base import FakeProviderMixin


class EchoFake(FakeProviderMixin):
    name = "echo"


client = AIClient(provider=EchoFake(), optimization=True)

r = client.generate(
    prompt="What is   the capital   of France?",
    system="You are a knowledgeable assistant.",
)
print("Response   :", r.content)
print("Tokens saved:", r.tokens_saved)
```

With a real provider simply change the provider name:

```python
client = AIClient(provider="openai", optimization=True)
```

and set `OPENAI_API_KEY` in your environment.

## CLI

```bash
ai-opticore init                # create opticore.json
ai-opticore config              # print active config
ai-opticore optimize --prompt "..." 
ai-opticore benchmark           # real measurements with fake provider
ai-opticore benchmark --provider openai
ai-opticore models              # list providers/models
ai-opticore hardware            # detect CPU/CUDA/ROCm
ai-opticore cache --clear
```

Every command supports `--help`.

## Configuration

`OptimizationConfig` fields (also reflected in `opticore.json` produced by
`ai-opticore init`):

| Field | Default | Meaning |
| --- | --- | --- |
| `safety_mode` | `safe` | `safe` / `balanced` / `aggressive` |
| `max_token_budget` | `4096` | context budget used by context optimization |
| `enable_token_optimization` | `true` | token analysis component |
| `enable_prompt_optimization` | `true` | prompt optimizer |
| `enable_context_optimization` | `true` | context optimizer |
| `enable_semantic_cache` | `true` | cache lookups before provider calls |
| `enable_model_routing` | `true` | routing component (no-op if no router attached) |
| `semantic_cache_threshold` | `0.90` | similarity threshold for cache hits |
| `cache_ttl_seconds` | `3600` | TTL of cached responses |
| `prioritize_recent` | `true` | keep recent messages when pruning |
| `log_prompts` | `false` | allow prompt text in logs |

Programmatic config:

```python
from opticore import Optimizer, OptimizationConfig, SafetyMode

config = OptimizationConfig(
    safety_mode=SafetyMode.BALANCED,
    max_token_budget=2048,
    enable_model_routing=False,
)
optimizer = Optimizer(config=config)
```