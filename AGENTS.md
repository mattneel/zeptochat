# Repository Guidelines

## Project Structure & Module Organization
Core Zig modules live in `src/`:
- `tokenizer.zig` – byte-pair encoder/decoder and fixtures
- `transformer.zig` – embeddings, attention, MLP, layer norms, and full backward pass
- `training.zig` – cross-entropy loss/gradients plus optimizer helpers

Unit and integration tests sit under `tests/` and are gathered in `tests/all_tests.zig` (gradient checks, transformer suites, training smoke tests). Example binaries belong in `examples/`, while mdBook content resides in `docs/`. Build artifacts belong only in `zig-out/` and should never be committed.

## Build, Test, and Development Commands
Bootstrap once with:
```bash
mise install
mise trust
```
Daily drivers:
- `mise run fmt` / `mise run fmt:fix` – enforce Zig formatting
- `mise run test` – run the full debug test suite (includes gradient checks and overfit training)
- `mise run test:opt` – ReleaseFast test sweep (mirrors CI)
- `mise run bench` – build+run performance benchmarks
- `mise run docs` – build the mdBook into `zig-out/docs`

All commands proxy `zig build …` via `build.zig`; do not call `zig test` directly unless adding new cases.

## Coding Style & Naming Conventions
Follow Zig 0.15.1 idioms: UpperCamelCase for types, snake_case for functions/vars, and avoid `usingnamespace`. Keep implementations linear (“TigerStyle”) with explicit control flow and minimal comptime tricks. Comments should explain *why* a choice exists (e.g., numerical stability, allocator lifetime) rather than paraphrasing code. Run `zig fmt` before every commit; CI rejects unformatted sources.

## Testing Guidelines
- Co-locate new tests with their module and import them from `tests/all_tests.zig`.
- For differentiable code, pair analytical gradients with numerical checks as seen in `tests/gradient_test.zig`.
- Keep runtime tight—tests currently complete in <10 s debug / <1 s ReleaseFast.
- When adding training behaviour, create an overfit smoke test that asserts loss reduction similar to `tests/training_test.zig`.
- Run both `mise run test` and `mise run test:opt` before submitting a PR; CI executes the same matrix on Linux/macOS/Windows.

## Commit & Pull Request Guidelines
Use descriptive branches (`feature/attention-backward`, `fix/training-loss`) and keep `master` history linear by rebasing before merge. Prefer Conventional Commit-style messages (`feat: add training overfit test`, `refactor: reuse matmul scratch buffers`). Each commit should compile and pass tests. PR descriptions must summarize scope, reference relevant sections of `SPEC.md`/`TODO.md`, and list validation commands (tests, benchmarks, docs). Attach logs only when they illuminate performance or numerical changes. Guardrail: do not merge without green CI and updated documentation when behaviour changes.
