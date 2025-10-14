# Roadmap

## Setup
- [ ] Initialize git repo
- [ ] Create directory structure
- [ ] Configure `build.zig` for Zig 0.15.1 (`root_module`)
- [ ] Finalize `.mise.toml` with required tools
- [ ] Author `README.md`
- [ ] Scaffold `docs/` for mdBook

## Phase 1 · CPU Baseline
- [ ] Tokenizer  
  - [ ] BPE encode  
  - [ ] BPE decode  
  - [ ] Vocab/merges loading  
  - [ ] Tests: round-trip, known examples
- [ ] Transformer  
  - [ ] Embeddings (token + position)  
  - [ ] Multi-head attention  
  - [ ] MLP with GELU  
  - [ ] Layer norm  
  - [ ] Forward pass  
  - [ ] Tests: shape checking, reference comparison
- [ ] Backpropagation  
  - [ ] Manual gradient computation  
  - [ ] Backward pass implementation  
  - [ ] Tests: gradient checking
- [ ] Optimizer (AdamW)  
  - [ ] Weight decay  
  - [ ] Parameter updates  
  - [ ] Tests: convergence on quadratic
- [ ] Training Loop  
  - [ ] Data loading (0.15.1 Reader API)  
  - [ ] Batch iteration  
  - [ ] Forward/backward/update loop  
  - [ ] Loss calculation  
  - [ ] Logging (buffered Writer API)  
  - [ ] Tests: overfit single batch

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
