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
3. Explore additional tasks, such as `mise run bench` or `mise run docs`, to build benchmarks and mdBook documentation (see `.mise.toml` for the full catalog).

## Project Structure

The expected layout (refer to `SPEC.md` for details):

```
src/            # Core Zig modules (tokenizer, transformer, training, etc.)
tests/          # Zig test harness importing focused suites
examples/       # Executable samples (CartPole, benchmarks)
docs/           # mdBook sources
zig-out/        # Build artifacts (ignored)
```

## Development Workflow

- **Language Version:** Zig 0.15.1; avoid deprecated constructs such as `usingnamespace`.
- **Formatting:** `zig fmt` is enforced via CI (`mise run fmt`).
- **Testing:** Follow test-driven development. Gradient checks and deterministic unit tests live under `tests/` and are aggregated by `tests/all_tests.zig`.
- **Git Flow:** Cut a named branch (`feature/tokenizer-merges`, `fix/reader-buffers`, etc.) for every change; no direct commits to `master`. Rebase onto `master` before opening a PR.
- **Build:** `zig build` uses `build.zig` with explicit modules and `root_module` semantics required by Zig 0.15.1.
- **Documentation:** mdBook-generated docs reside in `docs/` and are built via `mise run docs`.

## Contributing

Consult `AGENTS.md` for contributor guidelines, coding style expectations, and CI behaviours. The long-form implementation roadmap now resides in `TODO.md`; keep it updated as milestones are delivered. When in doubt about architecture or APIs, refer back to `SPEC.md` before introducing new abstractions.
