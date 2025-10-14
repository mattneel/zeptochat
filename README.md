# Zeptochat

Zeptochat is a transformer training playground implemented in pure Zig 0.15.1. The goal is to produce a CPU-first, WASM-ready stack that mirrors large-language-model workflows without external dependencies. This repository tracks the implementation plan defined in `SPEC.md` and treats documentation, tests, and code as a shared contract.

## Getting Started

1. Install [mise](https://github.com/jdx/mise) and activate the toolchain:
   ```bash
   mise install
   mise trust
   ```
2. Verify formatting and tests:
   ```bash
   mise run fmt
   mise run test
   ```
3. Explore additional tasks, such as `mise run test:opt`, `mise run bench`, or `mise run docs`, to exercise optimized builds, benchmarks, and mdBook output (see `.mise.toml` for the full catalog).

### Smoke test: end-to-end training

The debug test suite already includes an overfit experiment that drives the full forward/backward/optimizer loop and prints loss trends. To run it in isolation:
```bash
zig test tests/training_test.zig --dep transformer --dep training
```
To inspect the optimizer in ReleaseFast mode:
```bash
zig build test --summary all -Doptimize=ReleaseFast
```

## Project Structure

The expected layout (refer to `SPEC.md` for details):

```
src/            # Core Zig modules (tokenizer.zig, transformer.zig, training.zig, …)
tests/          # Gradient checks, transformer suites, training smoke tests
examples/       # Executable samples and benchmarks
docs/           # mdBook sources
zig-out/        # Build artifacts (ignored)
```

## Development Workflow

- **Language Version:** Zig 0.15.1; avoid deprecated constructs such as `usingnamespace`.
- **Formatting:** `zig fmt` is enforced via CI (`mise run fmt`).
- **Testing:** Follow test-driven development. Gradient checks, transformer suites, and the end-to-end overfit test live under `tests/` and are aggregated by `tests/all_tests.zig`.
- **Git Flow:** Cut a named branch (`feature/tokenizer-merges`, `fix/reader-buffers`, etc.) for every change; no direct commits to `master`. Rebase onto `master` before opening a PR.
- **Build:** `zig build` uses `build.zig` with explicit modules and `root_module` semantics required by Zig 0.15.1. The build graph exposes imports for `tokenizer`, `transformer`, and `training`.
- **Documentation:** mdBook-generated docs reside in `docs/` and are built via `mise run docs`.

## Contributing

Consult `AGENTS.md` for contributor guidelines, coding style expectations, and CI behaviours. The long-form implementation roadmap now resides in `TODO.md`; keep it updated as milestones are delivered. When in doubt about architecture or APIs, refer back to `SPEC.md` before introducing new abstractions.
