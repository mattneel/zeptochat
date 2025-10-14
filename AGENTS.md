# Repository Guidelines

## Project Structure & Module Organization
Core Zig code lives in `src/`, with focused modules such as `tokenizer.zig`, `transformer.zig`, `training.zig`, and `simd.zig`. Tests aggregate under `tests/` (see `tests/all_tests.zig`) so every new suite is imported there. mdBook sources sit in `docs/` (`docs/book.toml`, `docs/src/introduction.md`, etc.), and runnable samples stay in `examples/` (e.g., `examples/benchmark.zig`). Build artifacts land in `zig-out/`; keep only generated files there.

## Build, Test, and Development Commands
Use `mise install && mise trust` once to sync Zig 0.15.1 and mdBook. Day-to-day, rely on `mise run test` (wrapper for `zig build test`), `mise run bench` (benchmarks via `zig build bench`), and `mise run docs` for publishing docs. `zig build run -- train --config config.json` drives training binaries, while `zig build example` builds the CartPole smoke test; `./zig-out/bin/cartpole` should finish within the CI’s five-second probe.

## Coding Style & Naming Conventions
Always run `mise run fmt` or `zig fmt src/ tests/` before opening a PR; formatting is enforced by CI. Follow Zig idioms: structs and enums use UpperCamelCase, functions and variables stay snake_case, constants use ALL_CAPS only when representing true constants. Prefer linear, top-to-bottom “TigerStyle” implementations with explicit control flow, minimal comptime magic, and comments that justify decisions rather than restating code. Avoid `usingnamespace`; target the 0.15.1 APIs shown in `SPEC.md`.

## Testing Guidelines
Add focused tests beside the code they exercise and import them in `tests/all_tests.zig`. Maintain fast deterministic suites; gradient checks should mirror the numerical-vs-analytical pattern already described for attention layers. For behavior that requires data, overfit a single batch and assert monotonic loss drops. Run `zig build test -Doptimize=Debug` locally; CI also runs ReleaseFast builds on Linux, macOS (x86_64 and ARM64), and Windows, so keep platform-dependent logic guarded by feature checks.

## Commit & Pull Request Guidelines
With no public history yet, start fresh using Conventional Commit subjects (e.g., `feat: add tokenizer gradient tests`, `fix: flush checkpoint writer buffers`). Cut a descriptive branch (for example `feature/tokenizer-merges` or `chore/update-ci`) for each change and keep `master` fast-forward only by rebasing before PR. Group atomic changes per commit and ensure messages describe intent. Pull requests should summarize scope, link back to relevant `SPEC.md` sections, and include validation notes (tests, benchmarks, docs preview). Attach screenshots or logs only when they clarify regressions or performance shifts; otherwise keep the thread tight.
