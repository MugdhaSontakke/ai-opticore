"""Example: optimize a prompt and inspect token savings."""

from __future__ import annotations

from opticore import OptimizationConfig, Optimizer, SafetyMode

MESSY_PROMPT = (
    "   Please   write me a poem about   the ocean   \n\n"
    "A poem about the ocean. An ocean poem, please.\n\n"
    "Write a poem about the ocean waters and the sea and the tide. "
)

optimizer = Optimizer()  # defaults: SAFE mode

result = optimizer.optimize(
    prompt=MESSY_PROMPT,
    system="You are a helpful assistant.",
)

print(f"Original tokens:  {result.original_tokens}")
print(f"Optimized tokens: {result.optimized_tokens}")
print(f"Tokens saved:     {result.tokens_saved}")
print(f"Reduction:        {result.reduction_percent}%")
print(f"Optimizers run:   {', '.join(result.optimizers_run)}")
print(f"Optimized prompt: {result.optimized_prompt!r}")

# Try an aggressive variant to see the difference in behavior.
aggressive_config = OptimizationConfig(safety_mode=SafetyMode.AGGRESSIVE)
aggressive = Optimizer(config=aggressive_config)
outcome = aggressive.optimize(prompt=MESSY_PROMPT, system="You are a helpful assistant.")
print(f"\nAggressive saved: {outcome.tokens_saved} tokens "
      f"({outcome.reduction_percent}%). Warnings: {outcome.warnings_text}")
