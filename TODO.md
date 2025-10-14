# Roadmap

## Setup
- [x] Initialize git repo
- [x] Create directory structure
- [x] Configure `build.zig` for Zig 0.15.1 (`root_module`)
- [x] Finalize `.mise.toml` with required tools
- [x] Author `README.md`
- [x] Scaffold `docs/` for mdBook

## Phase 1 · CPU Baseline
- [x] Tokenizer  
  - [x] BPE encode  
  - [x] BPE decode  
  - [x] Vocab/merges loading  
  - [x] Tests: round-trip, known examples
- [x] Transformer  
  - [x] Embeddings (token + position)  
  - [x] Multi-head attention  
  - [x] MLP with GELU  
  - [x] Layer norm  
  - [x] Forward pass  
  - [x] Tests: shape checking, reference comparison
- [x] Backpropagation  
  - [x] Manual gradient computation  
  - [x] Backward pass implementation  
  - [x] Tests: gradient checking
- [ ] Optimizer (AdamW)  
  - [ ] Weight decay  
  - [ ] Parameter updates  
  - [ ] Tests: convergence on quadratic
- [ ] Training Loop  
  - [ ] Data loading (0.15.1 Reader API)  
  - [ ] Batch iteration  
  - [x] Forward/backward/update loop  
  - [x] Loss calculation  
  - [ ] Logging (buffered Writer API)  
  - [x] Tests: overfit single batch

## Phase 2 · SIMD
- [ ] Benchmark harness
- [ ] Scalar implementations (baseline)
- [ ] AVX2 implementations
- [ ] NEON implementations (ARM)
- [ ] Comptime dispatch (0.15.1 style)
- [ ] Tests: SIMD matches scalar
- [ ] Performance measurements

## Phase 3 · Multi-threading
- [ ] Thread pool via `std.Thread`
- [ ] Parallel matmul
- [ ] Thread-safe gradient accumulation
- [ ] Tests: multi-threaded matches single-thread
- [ ] Scaling benchmarks

## Phase 4 · Full Pipeline
- [ ] Dataset preparation scripts
- [ ] Efficient data loading
- [ ] Checkpoint save/load
- [ ] Evaluation loop
- [ ] Logging with 0.15.1 Writer API
- [ ] Generation (sampling)
- [ ] Train small model
- [ ] Validate generation quality

## Phase 5 · WASM
- [ ] WASM build configuration
- [ ] JavaScript bindings
- [ ] Memory management
- [ ] Web UI (HTML/CSS/JS)
- [ ] IndexedDB integration
- [ ] Web Workers
- [ ] Deployment
- [ ] Browser testing

## Phase 6 · Documentation
- [ ] Complete mdBook content
- [ ] Code comments
- [ ] README examples
- [ ] CONTRIBUTING guide
- [ ] Performance documentation
- [ ] Blog post
