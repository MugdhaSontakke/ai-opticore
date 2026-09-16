# Adding a new optimizer

Every optimizer is a pluggable component that transforms an `OptimizerRequest`
and reports token impact. New optimizers should be added without modifying the
rest of the codebase.

## Step 1: Create the module

```text
src/opticore/optimizers/yourname.py
```

## Step 2: Implement the contract

```python
"""YourName optimizer — short description."""

from __future__ import annotations

from typing import Any

from opticore.core.config import OptimizationConfig
from opticore.core.interfaces import OptimizerRequest, OptimizerResult
from opticore.optimizers.base import BaseOptimizer
from opticore.optimizers.token import Tokenizer, count_tokens


class YourNameOptimizer(BaseOptimizer):

    name = "yourname"
    order = 50          # lower runs first; pick a unique number

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or Tokenizer()

    def optimize(self, request: OptimizerRequest, config: OptimizationConfig) -> OptimizerResult:
        original = count_tokens(request, self.tokenizer)

        new_request = OptimizerRequest(
            prompt=your_transform(request.prompt),
            system=request.system,
            messages=request.messages,
            max_tokens=request.max_tokens,
            model=request.model,
            metadata=dict(request.metadata),
        )

        optimized = count_tokens(new_request, self.tokenizer)
        saved = max(0, original - optimized)

        return OptimizerResult(
            optimized_request=new_request,
            original_request=request,
            original_tokens=original,
            optimized_tokens=optimized,
            optimizer_name=self.name,
            optimizers_run=[self.name],
            tokens_saved=saved,
            reduction_percent=(saved / original * 100.0) if original else 0.0,
            metadata={},
        )
```

## Key rules

- **Never mutate `request`**. Always return a new `OptimizerRequest`.
- **Use real token counts**, not character length.
- **Document what each mode does** (SAFE/BALANCED/AGGRESSIVE).
- **Every new optimizer gets its own unit tests** in `tests/`.

## Step 3: Register in the pipeline

Add the optimizer to `build_default_pipeline()` in `core/pipeline.py` and
choose a stable pipeline `order` so the optimizer slot is deterministic.

Add a config flag in `OptimizationConfig` and check it in
`_is_enabled()` if the optimizer should be disableable.

## Step 4: Tests

Create `tests/test_yourname.py` covering at minimum:

- Basic transformation produces fewer tokens or equal.
- Original request is not mutated.
- Each relevant safety mode does what it claims.
- Token counts are accurate.

## Example tests

```python
def test_optimization_does_not_mutate_input() -> None:
    from opticore.core.interfaces import OptimizerRequest

    optimizer = YourNameOptimizer()
    request = OptimizerRequest(prompt="test", system="sys")
    prompt_copy = request.prompt
    optimizer.optimize(request, OptimizationConfig())
    assert request.prompt == prompt_copy
```