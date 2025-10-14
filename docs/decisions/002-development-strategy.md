# Decision Record 002 — Vertical Slice Strategy

- **Date:** 2025-10-14
- **Status:** Accepted
- **Context:** After finishing the tokenizer, we must choose a path for Phase 2 (transformer) and beyond. Options included TigerStyle linear sequence, SIMD-first optimisation, or a vertical slice that prioritises an end-to-end training loop.

## Decision
- Adopt a **vertical slice** approach:
  1. Implement scalar transformer forward pass with exhaustive tests.
  2. Add manual backward pass and gradient checks.
  3. Build a toy training loop that overfits a tiny dataset (loss → 0).
  4. Optimise (SIMD, threading) only after end-to-end correctness.

## Rationale
- Fast feedback uncovers architectural bugs before deep optimisation.
- Early convergence signal keeps momentum and guides prioritisation.
- Profiling after a working pipeline identifies true hot spots.
- Mirrors TigerStyle values: linear, explicit, data-driven decisions.

## Consequences
- Near-term focus shifts to transformer correctness rather than SIMD.
- Training data loader and performance work can run in parallel later.
- Branching plan: `feature/transformer-forward` → `feature/transformer-backward` → `feature/training-loop`.
- Once training slice lands, we will open PR and only then pursue SIMD/multithreading phases.
