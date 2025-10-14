# Decision Record 001 — Tokenizer Implementation Scope

- **Date:** 2025-10-14
- **Status:** Accepted
- **Context:** Phase 1 required a production-ready tokenizer ahead of transformer work. GPT-2 compatibility was desirable but full byte-to-Unicode remapping would add complexity without blocking MVP.

## Outcomes
- Implemented pure Zig tokenizer with:
  - File-backed `init(allocator, vocab_path, merges_path)` plus `initEmpty` for tests.
  - Greedy BPE merges, rank tie-breaking, and cascading collapse.
  - Byte-level fallback for UTF-8 sequences (no GPT-2 byte encoder shim yet).
  - Special token registration via `registerSpecialToken`, tracking `vocabSize`, `lookupTokenId`.
  - Fixture loader for trimmed GPT-2 vocab/merges (`tests/fixtures/gpt2_mini`).
- Added verbose test runner and comprehensive unit coverage (ASCII, UTF-8, merges, fixtures).

## Rationale
- Vertical slice demands encode→decode round-trip correctness over exact GPT-2 string surface form.
- Keeping ASCII byte IDs simplifies early integration with data loader and transformer embeddings.
- Special tokens (`<|eot|>`, `<|pad|>`) are required for training boundaries; now supported.
- GPT-2 byte encoder can be revisited if future compatibility demands identical merges.

## Consequences
- Token IDs align with GPT-2 for included fixtures; byte-level fallback ensures decoding fidelity.
- Further tokenizer tweaks (byte encoder, regex pre-tokenization) can happen post-transformer.
- Documentation updated (`SPEC.md`, ADR) and tests fail fast if vocabulary regressions occur.
