# Zeptochat: Implementation Specification for Zig 0.15.1

## Project Overview

**Zeptochat** is a minimal, hackable implementation of transformer-based language model training in pure Zig 0.15.1. It's inspired by Karpathy's nanochat but targets CPU-first execution with WASM browser deployment as the ultimate goal. The name "zepto" (10^-21) represents being orders of magnitude more minimal than "nano" (10^-9).

**Tagline:** "The best ChatGPT that Zig can compile."

**Philosophy:** 
- Part of the 0wntelligence ecosystem (own your training + own your inference)
- Complete AI sovereignty from training to deployment
- Accessible to anyone with consumer hardware
- No framework bloat, no cloud dependency

## Critical Zig 0.15.1 Breaking Changes

### Language Changes You Must Know

**1. `usingnamespace` is DELETED**
- Do NOT use `pub usingnamespace` anywhere
- Use explicit declarations or switch-based conditionals instead
- Feature detection: use `if (@TypeOf(foo) == void)` pattern

**2. New I/O System ("Writergate")**
- `std.io.GenericWriter` → `std.Io.Writer` (concrete, non-generic)
- `std.io.GenericReader` → `std.Io.Reader` (concrete, non-generic)
- `std.fs.File.writer()` → `std.fs.File.writer(&buffer)` (requires buffer)
- `std.fs.File.reader()` → `std.fs.File.reader(&buffer)` (requires buffer)
- BufferedWriter/BufferedReader deleted - use buffer parameter instead
- Writer/Reader now have buffers ABOVE the vtable, not in implementation

**3. Format String Changes**
- `"{}"` with custom format methods now errors - use `"{f}"` explicitly
- Format methods signature changed:
  ```zig
  // OLD (0.14.x)
  pub fn format(
      this: @This(),
      comptime format_string: []const u8,
      options: std.fmt.FormatOptions,
      writer: anytype,
  ) !void
  
  // NEW (0.15.1)
  pub fn format(this: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void
  ```
- Unicode formatting removed - ASCII/bytes only

**4. Build System**
- `root_source_file` field DELETED (removed in 0.14, enforced in 0.15)
- Must use `root_module` field:
  ```zig
  // OLD - COMPILE ERROR
  const exe = b.addExecutable(.{
      .name = "foo",
      .root_source_file = b.path("src/main.zig"),
  });
  
  // NEW - CORRECT
  const exe = b.addExecutable(.{
      .name = "foo",
      .root_module = b.createModule(.{
          .root_source_file = b.path("src/main.zig"),
      }),
  });
  ```

**5. ArrayList Changes**
- `std.ArrayList` → `std.ArrayListUnmanaged` (default is now unmanaged)
- `std.ArrayListAligned` → `std.array_list.AlignedManaged` (deprecated)
- Old managed versions deprecated, will be removed

**6. Inline Assembly**
- Clobbers now typed, not strings:
  ```zig
  // OLD
  : "rcx", "r11"
  
  // NEW
  : .{ .rcx = true, .r11 = true }
  ```

**7. BoundedArray DELETED**
- Replace with `std.ArrayListUnmanaged` + stack buffer
- Use `initBuffer()` for stack allocation

**8. Standard Library Deletions**
- `std.fifo.LinearFifo` - deleted
- `std.RingBuffer` - deleted
- `std.io.BufferedWriter` - deleted
- `std.io.BufferedReader` - deleted
- `std.io.CountingWriter` - deleted
- `std.io.SeekableStream` - deleted
- `std.compress.flate` - completely reworked API

## Core Principles

### Development Philosophy (TigerStyle + Data-Driven + TDD + DDD)

**TigerStyle Principles:**
- Linear, top-to-bottom code flow
- No unnecessary libraries (Zig stdlib only)
- No classes, just functions and data
- No frameworks or magic
- Inline critical paths
- Comments explain why, not what
- Copy-paste over DRY when it clarifies
- Explicit over clever

**Data-Driven Development (Zig style):**
- Measure everything before optimizing
- Profile with real numbers
- Benchmark variants, choose based on data
- "Show me the numbers" for architectural decisions

**Test-Driven Development:**
- Write tests first (expected behavior)
- Implement until tests pass
- Gradient checks, loss curves, eval metrics as tests
- Tests serve as executable specifications

**Domain-Driven Design (for mental modeling):**
- Model domain concepts explicitly (Tokenizer, Transformer, Optimizer)
- Use ML terminology consistently (attention, embeddings, logits)
- Bounded contexts (training vs inference as separate domains)
- Rich domain models, not anemic data structures

**Documentation as Contract:**
- Tests + Documentation = Single source of truth
- Doc comments in code
- Manual mdBook for guides
- Test aggregator modules for organization

### Linear Development Sequence

Build one component at a time, validate, then move to next:

1. **Tokenizer** (works, tested)
2. **Transformer forward pass** (validated against reference)
3. **Backpropagation** (correct gradients via gradient checking)
4. **Training loop** (convergence on toy data)
5. **SIMD optimization** (measured speedup)
6. **Multi-threading** (verified correctness + performance)
7. **WASM port** (once native works perfectly)

**Never:** Build everything simultaneously with beautiful abstractions.

## Technical Architecture

### Technology Stack

**Core:**
- **Zig 0.15.1** - Systems language, comptime, SIMD, x86 backend default in Debug
- **No external dependencies** - stdlib only for core training
- **mdBook** - Manual documentation (not automated doctests)
- **mise** - Development environment + task runner

**Target Platforms (via comptime):**
- x86-64 with AVX-512 (server/desktop) - **SELF-HOSTED BACKEND DEFAULT IN DEBUG**
- x86-64 with AVX2 (laptops, older desktops) - **SELF-HOSTED BACKEND DEFAULT IN DEBUG**
- AArch64 with NEON (Apple Silicon, ARM servers) - LLVM backend (for now)
- WASM with SIMD (browser deployment)
- Scalar fallback (universal compatibility)

### Repository Structure

```
zeptochat/
├── src/
│   ├── main.zig              # Entry point, CLI
│   ├── tokenizer.zig         # BPE tokenization
│   ├── transformer.zig       # Model architecture
│   ├── training.zig          # Training loop
│   ├── optimizer.zig         # AdamW optimizer
│   ├── data.zig              # Data loading
│   ├── simd.zig              # SIMD operations
│   └── checkpoint.zig        # Save/load model
├── tests/
│   ├── all_tests.zig         # Test aggregator
│   ├── tokenizer_test.zig
│   ├── transformer_test.zig
│   ├── training_test.zig
│   └── gradient_test.zig
├── docs/
│   ├── book.toml
│   ├── SUMMARY.md
│   └── src/
│       ├── introduction.md
│       ├── architecture.md
│       ├── training.md
│       └── deployment.md
├── examples/
│   ├── train_tiny.zig        # Overfit single batch
│   └── benchmark.zig         # Performance testing
├── .mise.toml                # Dev environment
├── build.zig                 # Build configuration
├── README.md
└── LICENSE                   # MIT
```

### Comptime SIMD Dispatch (0.15.1 Compatible)

**Core Pattern:**
```zig
const builtin = @import("builtin");
const std = @import("std");

// Feature detection
const has_avx512 = comptime blk: {
    const features = builtin.cpu.features;
    break :blk std.Target.x86.featureSetHas(features, .avx512f);
};

const has_avx2 = comptime blk: {
    const features = builtin.cpu.features;
    break :blk std.Target.x86.featureSetHas(features, .avx2);
};

const has_neon = comptime blk: {
    const features = builtin.cpu.features;
    break :blk std.Target.aarch64.featureSetHas(features, .neon);
};

pub fn matmul(a: []const f32, b: []const f32, out: []f32, M: usize, N: usize, K: usize) void {
    if (comptime has_avx512) {
        matmul_avx512(a, b, out, M, N, K);
    } else if (comptime has_avx2) {
        matmul_avx2(a, b, out, M, N, K);
    } else if (comptime has_neon) {
        matmul_neon(a, b, out, M, N, K);
    } else {
        matmul_scalar(a, b, out, M, N, K);
    }
}
```

**Benefits:**
- Zero runtime overhead (branches eliminated at compile time)
- Single codebase for all platforms
- Optimal code generation for target CPU
- Users just `zig build` and get best performance

## Component Specifications

### 1. Tokenizer (BPE)

**Purpose:** Byte Pair Encoding tokenization (similar to GPT-2/3)

**Interface:**
```zig
pub const Tokenizer = struct {
    vocab: []const []const u8,
    merges: []const Merge,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, vocab_file: []const u8, merges_file: []const u8) !Tokenizer;
    pub fn deinit(self: *Tokenizer) void;
    pub fn encode(self: *Tokenizer, text: []const u8) ![]u32;
    pub fn decode(self: *Tokenizer, tokens: []const u32) ![]u8;
};
```

**Testing:**
- Round-trip test: `decode(encode(text)) == text`
- Known examples from GPT-2 tokenizer
- Unicode handling (emoji, international characters)

**Implementation Notes:**
- Start with simple BPE (no regex pre-tokenization initially)
- Use `std.StringHashMap` for vocab lookup
- Allocate with provided allocator, never implicit

### 2. Transformer Architecture

**Model Specification:**
- GPT-2 style decoder-only transformer
- Configurable depth, width, context length
- Positional embeddings (learned)
- Multi-head attention with causal masking
- MLP with GELU activation
- Layer normalization

**Core Types (0.15.1 Compatible):**
```zig
const std = @import("std");

pub const ModelConfig = struct {
    vocab_size: usize,
    context_length: usize,
    d_model: usize,          // Model dimension
    n_heads: usize,          // Attention heads
    n_layers: usize,         // Transformer blocks
    dropout: f32,
};

pub const Transformer = struct {
    config: ModelConfig,
    allocator: std.mem.Allocator,
    token_embeddings: []f32,     // [vocab_size, d_model]
    position_embeddings: []f32,  // [context_length, d_model]
    layers: []TransformerBlock,
    ln_final: LayerNorm,
    lm_head: []f32,             // [d_model, vocab_size]
    
    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !Transformer;
    pub fn deinit(self: *Transformer) void;
    pub fn forward(self: *Transformer, tokens: []const u32) ![]f32; // Returns logits
    pub fn backward(self: *Transformer, grad_logits: []const f32) !void;
    pub fn zeroGrad(self: *Transformer) void;
};

pub const TransformerBlock = struct {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    mlp: MLP,
    
    pub fn forward(self: *TransformerBlock, hidden: []f32) ![]f32;
    pub fn backward(self: *TransformerBlock, grad: []const f32) !void;
};

pub const LayerNorm = struct {
    weight: []f32,
    bias: []f32,
    eps: f32 = 1e-5,
    
    pub fn forward(self: *LayerNorm, x: []f32) void;
    pub fn backward(self: *LayerNorm, grad: []const f32) !void;
};

pub const MultiHeadAttention = struct {
    config: ModelConfig,
    w_q: []f32,
    w_k: []f32,
    w_v: []f32,
    w_o: []f32,
    
    pub fn forward(self: *MultiHeadAttention, x: []const f32) ![]f32;
    pub fn backward(self: *MultiHeadAttention, grad: []const f32) !void;
};

pub const MLP = struct {
    w1: []f32,
    b1: []f32,
    w2: []f32,
    b2: []f32,
    
    pub fn forward(self: *MLP, x: []const f32) ![]f32;
    pub fn backward(self: *MLP, grad: []const f32) !void;
};
```

**Forward Pass (Linear Top-to-Bottom):**
```zig
pub fn forward(self: *Transformer, tokens: []const u32) ![]f32 {
    const batch_size = tokens.len;
    const d_model = self.config.d_model;
    
    // 1. Embed tokens and positions
    var hidden = try self.allocator.alloc(f32, batch_size * d_model);
    errdefer self.allocator.free(hidden);
    
    for (tokens, 0..) |token, i| {
        const token_idx = token * d_model;
        const pos_idx = i * d_model;
        const hidden_idx = i * d_model;
        
        // hidden[i] = token_embed[token] + pos_embed[i]
        for (0..d_model) |j| {
            hidden[hidden_idx + j] = self.token_embeddings[token_idx + j] + 
                                     self.position_embeddings[pos_idx + j];
        }
    }
    
    // 2. Apply transformer blocks
    for (self.layers) |*layer| {
        const new_hidden = try layer.forward(hidden);
        self.allocator.free(hidden);
        hidden = new_hidden;
    }
    
    // 3. Final layer norm
    self.ln_final.forward(hidden);
    
    // 4. Project to vocabulary (logits)
    const vocab_size = self.config.vocab_size;
    var logits = try self.allocator.alloc(f32, batch_size * vocab_size);
    
    // Matrix multiply: hidden @ lm_head -> logits
    matmul(hidden, self.lm_head, logits, batch_size, vocab_size, d_model);
    
    self.allocator.free(hidden);
    return logits;
}
```

**Testing:**
- Gradient checking (numerical vs analytical gradients)
- Forward pass matches reference implementation
- Attention mask correctness (causal masking)
- Shape checking at each layer
- Overfit single batch (loss should go to near-zero)

### 3. New I/O System (0.15.1 Writergate Migration)

**CRITICAL: Old Writer/Reader Patterns Are DELETED**

**OLD CODE (0.14.x) - DOES NOT COMPILE:**
```zig
// THIS IS WRONG - DO NOT USE
const stdout = std.io.getStdOut().writer();
try stdout.print("Hello, world!\n", .{});
```

**NEW CODE (0.15.1) - CORRECT:**
```zig
// User provides buffer, writer uses it
var stdout_buffer: [4096]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout: *std.Io.Writer = &stdout_writer.interface;

try stdout.print("Hello, world!\n", .{});
try stdout.flush(); // DON'T FORGET TO FLUSH!
```

**File Reading (0.15.1):**
```zig
const std = @import("std");

pub fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    
    // Provide buffer for reader
    var read_buffer: [8192]u8 = undefined;
    var file_reader = file.reader(&read_buffer);
    const reader: *std.Io.Reader = &file_reader.interface;
    
    // Read all content
    var content = std.ArrayListUnmanaged(u8){};
    errdefer content.deinit(allocator);
    
    while (true) {
        const chunk = try reader.readAtLeast(read_buffer.len);
        if (chunk.len == 0) break;
        try content.appendSlice(allocator, chunk);
    }
    
    return content.toOwnedSlice(allocator);
}
```

**Formatted Output Pattern:**
```zig
const std = @import("std");

pub fn logTraining(step: usize, loss: f32) !void {
    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr: *std.Io.Writer = &stderr_writer.interface;
    
    try stderr.print("Step {d}, Loss: {d:.4}\n", .{step, loss});
    try stderr.flush();
}
```

### 4. SIMD Operations (0.15.1 Compatible)

**Critical Operations to Optimize:**
- Matrix multiplication (matmul)
- Element-wise operations (add, multiply, GELU)
- Softmax
- Layer normalization
- Attention computation

**SIMD Module Structure:**
```zig
// src/simd.zig
const std = @import("std");
const builtin = @import("builtin");

// Feature detection (comptime)
const has_avx512 = comptime blk: {
    const features = builtin.cpu.features;
    break :blk std.Target.x86.featureSetHas(features, .avx512f);
};

const has_avx2 = comptime blk: {
    const features = builtin.cpu.features;
    break :blk std.Target.x86.featureSetHas(features, .avx2);
};

const has_neon = comptime blk: {
    if (builtin.cpu.arch != .aarch64) break :blk false;
    const features = builtin.cpu.features;
    break :blk std.Target.aarch64.featureSetHas(features, .neon);
};

// Scalar baseline (always works)
pub fn matmul_scalar(
    a: []const f32,
    b: []const f32,
    out: []f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

// AVX2 optimized (256-bit)
pub fn matmul_avx2(
    a: []const f32,
    b: []const f32,
    out: []f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    // Implementation using AVX2 intrinsics
    // @Vector(8, f32) for 256-bit SIMD
    _ = a;
    _ = b;
    _ = out;
    _ = M;
    _ = N;
    _ = K;
    @compileError("TODO: implement AVX2 matmul");
}

// AVX-512 optimized (512-bit)
pub fn matmul_avx512(
    a: []const f32,
    b: []const f32,
    out: []f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    // Implementation using AVX-512 intrinsics
    // @Vector(16, f32) for 512-bit SIMD
    _ = a;
    _ = b;
    _ = out;
    _ = M;
    _ = N;
    _ = K;
    @compileError("TODO: implement AVX-512 matmul");
}

// NEON optimized (ARM)
pub fn matmul_neon(
    a: []const f32,
    b: []const f32,
    out: []f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    // Implementation using NEON intrinsics
    _ = a;
    _ = b;
    _ = out;
    _ = M;
    _ = N;
    _ = K;
    @compileError("TODO: implement NEON matmul");
}

// Public API with comptime dispatch
pub fn matmul(
    a: []const f32,
    b: []const f32,
    out: []f32,
    M: usize,
    N: usize,
    K: usize,
) void {
    if (comptime has_avx512) {
        matmul_avx512(a, b, out, M, N, K);
    } else if (comptime has_avx2) {
        matmul_avx2(a, b, out, M, N, K);
    } else if (comptime has_neon) {
        matmul_neon(a, b, out, M, N, K);
    } else {
        matmul_scalar(a, b, out, M, N, K);
    }
}

// Element-wise operations
pub fn add_scalar(a: []const f32, b: []const f32, out: []f32) void {
    for (a, b, out) |a_val, b_val, *out_val| {
        out_val.* = a_val + b_val;
    }
}

pub fn gelu_scalar(x: []const f32, out: []f32) void {
    const sqrt_2_over_pi = @sqrt(2.0 / std.math.pi);
    for (x, out) |x_val, *out_val| {
        const cdf = 0.5 * (1.0 + std.math.tanh(sqrt_2_over_pi * (x_val + 0.044715 * x_val * x_val * x_val)));
        out_val.* = x_val * cdf;
    }
}
```

**Benchmarking (0.15.1):**
```zig
// examples/benchmark.zig
const std = @import("std");
const simd = @import("simd");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const M = 1024;
    const N = 1024;
    const K = 1024;
    
    const a = try allocator.alloc(f32, M * K);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, K * N);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, M * N);
    defer allocator.free(out);
    
    // Initialize with random data
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (a) |*val| val.* = random.float(f32);
    for (b) |*val| val.* = random.float(f32);
    
    // Benchmark scalar
    const start_scalar = std.time.nanoTimestamp();
    simd.matmul_scalar(a, b, out, M, N, K);
    const end_scalar = std.time.nanoTimestamp();
    
    // Benchmark SIMD
    const start_simd = std.time.nanoTimestamp();
    simd.matmul(a, b, out, M, N, K); // Dispatches to best
    const end_simd = std.time.nanoTimestamp();
    
    const scalar_ms = @as(f64, @floatFromInt(end_scalar - start_scalar)) / 1_000_000.0;
    const simd_ms = @as(f64, @floatFromInt(end_simd - start_simd)) / 1_000_000.0;
    const speedup = scalar_ms / simd_ms;
    
    var stderr_buffer: [512]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr: *std.Io.Writer = &stderr_writer.interface;
    
    try stderr.print("Scalar: {d:.2}ms\n", .{scalar_ms});
    try stderr.print("SIMD: {d:.2}ms\n", .{simd_ms});
    try stderr.print("Speedup: {d:.2}x\n", .{speedup});
    try stderr.flush();
}
```

### 5. Training Loop (0.15.1 Compatible)

**Training Configuration:**
```zig
pub const TrainingConfig = struct {
    batch_size: usize,
    sequence_length: usize,
    num_epochs: usize,
    learning_rate: f32,
    weight_decay: f32,
    grad_clip: f32,
    checkpoint_every: usize,  // Save every N steps
    eval_every: usize,        // Evaluate every N steps
};
```

**Training Loop (Linear, Explicit, 0.15.1):**
```zig
const std = @import("std");

pub fn train(
    model: *Transformer,
    optimizer: *AdamW,
    dataset: *Dataset,
    config: TrainingConfig,
    allocator: std.mem.Allocator,
) !void {
    var step: usize = 0;
    
    // Setup logging
    var log_buffer: [1024]u8 = undefined;
    var log_writer = std.fs.File.stderr().writer(&log_buffer);
    const log: *std.Io.Writer = &log_writer.interface;
    
    for (0..config.num_epochs) |epoch| {
        var batch_iter = dataset.iterator(config.batch_size, config.sequence_length);
        
        while (batch_iter.next()) |batch| {
            // 1. Forward pass
            const logits = try model.forward(batch.tokens);
            defer allocator.free(logits);
            
            const loss = crossEntropyLoss(logits, batch.targets, batch.tokens.len, model.config.vocab_size);
            
            // 2. Backward pass
            const grad_logits = try crossEntropyGrad(
                allocator,
                logits,
                batch.targets,
                batch.tokens.len,
                model.config.vocab_size,
            );
            defer allocator.free(grad_logits);
            
            try model.backward(grad_logits);
            
            // 3. Clip gradients
            clipGradients(model, config.grad_clip);
            
            // 4. Optimizer step
            try optimizer.step(model, config.learning_rate);
            
            // 5. Zero gradients
            model.zeroGrad();
            
            // 6. Logging
            if (step % 100 == 0) {
                try log.print("Epoch {d}, Step {d}, Loss: {d:.4}\n", .{epoch, step, loss});
                try log.flush();
            }
            
            // 7. Checkpointing
            if (step % config.checkpoint_every == 0) {
                try saveCheckpoint(model, optimizer, step, allocator);
            }
            
            // 8. Evaluation
            if (step % config.eval_every == 0) {
                const eval_loss = try evaluate(model, dataset.validation_set, allocator);
                try log.print("Eval Loss: {d:.4}\n", .{eval_loss});
                try log.flush();
            }
            
            step += 1;
        }
    }
}

fn crossEntropyLoss(logits: []const f32, targets: []const u32, batch_size: usize, vocab_size: usize) f32 {
    var total_loss: f32 = 0;
    
    for (0..batch_size) |i| {
        const target = targets[i];
        const logit_start = i * vocab_size;
        
        // Softmax denominator
        var max_logit: f32 = -std.math.inf(f32);
        for (0..vocab_size) |j| {
            max_logit = @max(max_logit, logits[logit_start + j]);
        }
        
        var sum_exp: f32 = 0;
        for (0..vocab_size) |j| {
            sum_exp += @exp(logits[logit_start + j] - max_logit);
        }
        
        // Cross-entropy for this example
        const target_logit = logits[logit_start + target];
        total_loss -= target_logit - max_logit - @log(sum_exp);
    }
    
    return total_loss / @as(f32, @floatFromInt(batch_size));
}

fn crossEntropyGrad(
    allocator: std.mem.Allocator,
    logits: []const f32,
    targets: []const u32,
    batch_size: usize,
    vocab_size: usize,
) ![]f32 {
    var grad = try allocator.alloc(f32, logits.len);
    
    for (0..batch_size) |i| {
        const target = targets[i];
        const logit_start = i * vocab_size;
        
        // Compute softmax
        var max_logit: f32 = -std.math.inf(f32);
        for (0..vocab_size) |j| {
            max_logit = @max(max_logit, logits[logit_start + j]);
        }
        
        var sum_exp: f32 = 0;
        for (0..vocab_size) |j| {
            sum_exp += @exp(logits[logit_start + j] - max_logit);
        }
        
        // Gradient: softmax - one_hot
        for (0..vocab_size) |j| {
            const softmax = @exp(logits[logit_start + j] - max_logit) / sum_exp;
            const one_hot: f32 = if (j == target) 1.0 else 0.0;
            grad[logit_start + j] = (softmax - one_hot) / @as(f32, @floatFromInt(batch_size));
        }
    }
    
    return grad;
}

fn clipGradients(model: *Transformer, max_norm: f32) void {
    var total_norm: f32 = 0;
    
    // Compute total gradient norm
    // (iterate through all parameters)
    _ = model;
    
    // Clip if necessary
    if (total_norm > max_norm) {
        const scale = max_norm / total_norm;
        // Scale all gradients
        _ = scale;
    }
}
```

**Testing:**
- Overfit single batch (loss → 0)
- Learning rate sensitivity
- Gradient clipping prevents explosion
- Checkpoint save/load preserves state exactly
- Loss decreases over time on real data

### 6. Optimizer (AdamW) - 0.15.1

**Implementation:**
```zig
const std = @import("std");

pub const AdamW = struct {
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    m: []f32,  // First moment (mean)
    v: []f32,  // Second moment (variance)
    step_count: usize = 0,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, num_params: usize) !AdamW {
        const m = try allocator.alloc(f32, num_params);
        errdefer allocator.free(m);
        
        const v = try allocator.alloc(f32, num_params);
        errdefer allocator.free(v);
        
        @memset(m, 0);
        @memset(v, 0);
        
        return AdamW{
            .m = m,
            .v = v,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *AdamW) void {
        self.allocator.free(self.m);
        self.allocator.free(self.v);
    }
    
    pub fn step(self: *AdamW, model: *Transformer, lr: f32, weight_decay: f32) !void {
        self.step_count += 1;
        const t = @as(f32, @floatFromInt(self.step_count));
        
        // Bias correction
        const bias_correction1 = 1.0 - std.math.pow(f32, self.beta1, t);
        const bias_correction2 = 1.0 - std.math.pow(f32, self.beta2, t);
        const lr_t = lr * @sqrt(bias_correction2) / bias_correction1;
        
        // Update all parameters
        // This is a simplified version - real implementation needs to iterate
        // through all model parameters (token_embeddings, position_embeddings,
        // layer weights, etc.)
        _ = model;
        _ = lr_t;
        _ = weight_decay;
        
        // For each parameter:
        // 1. m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
        // 2. v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
        // 3. param[i] -= lr * weight_decay * param[i]  (weight decay)
        // 4. param[i] -= lr_t * m[i] / (sqrt(v[i]) + eps)  (Adam update)
    }
};
```

### 7. Data Loading (0.15.1 Compatible)

**Dataset Interface:**
```zig
const std = @import("std");

pub const Dataset = struct {
    data: []const u8,        // Raw text data
    tokenized: []const u32,  // Pre-tokenized
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, data_path: []const u8, tokenizer: *Tokenizer) !Dataset {
        // Read file
        const file = try std.fs.cwd().openFile(data_path, .{});
        defer file.close();
        
        const data = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        errdefer allocator.free(data);
        
        // Tokenize
        const tokenized = try tokenizer.encode(data);
        
        return Dataset{
            .data = data,
            .tokenized = tokenized,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Dataset) void {
        self.allocator.free(self.data);
        self.allocator.free(self.tokenized);
    }
    
    pub fn iterator(self: *Dataset, batch_size: usize, seq_len: usize) BatchIterator {
        return BatchIterator{
            .dataset = self,
            .batch_size = batch_size,
            .seq_len = seq_len,
            .position = 0,
        };
    }
    
    pub const validation_set: *Dataset = undefined; // TODO: implement split
};

pub const Batch = struct {
    tokens: []const u32,   // [batch_size * seq_len]
    targets: []const u32,  // [batch_size * seq_len] (shifted by 1)
};

pub const BatchIterator = struct {
    dataset: *Dataset,
    batch_size: usize,
    seq_len: usize,
    position: usize,
    
    pub fn next(self: *BatchIterator) ?Batch {
        const total_tokens = self.batch_size * self.seq_len;
        
        if (self.position + total_tokens + 1 > self.dataset.tokenized.len) {
            return null;
        }
        
        const tokens = self.dataset.tokenized[self.position..][0..total_tokens];
        const targets = self.dataset.tokenized[self.position + 1..][0..total_tokens];
        
        self.position += total_tokens;
        
        return Batch{
            .tokens = tokens,
            .targets = targets,
        };
    }
};
```

### 8. Multi-Threading (0.15.1)

**Use Zig's stdlib threading:**
```zig
const std = @import("std");

pub fn parallelMatmul(
    allocator: std.mem.Allocator,
    a: []const f32,
    b: []const f32,
    out: []f32,
    M: usize,
    N: usize,
    K: usize,
    num_threads: usize,
) !void {
    var threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);
    
    const rows_per_thread = M / num_threads;
    
    for (threads, 0..) |*thread, i| {
        const start_row = i * rows_per_thread;
        const end_row = if (i == num_threads - 1) M else (i + 1) * rows_per_thread;
        
        thread.* = try std.Thread.spawn(.{}, matmulRange, .{
            a, b, out, start_row, end_row, N, K
        });
    }
    
    for (threads) |thread| {
        thread.join();
    }
}

fn matmulRange(
    a: []const f32,
    b: []const f32,
    out: []f32,
    start_row: usize,
    end_row: usize,
    N: usize,
    K: usize,
) void {
    for (start_row..end_row) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}
```

**Testing:**
- Verify multi-threaded results match single-threaded
- Benchmark speedup (should scale with cores)
- No race conditions (run with thread sanitizer)

### 9. WASM Target (0.15.1)

**Build for WASM:**
```bash
zig build -Dtarget=wasm32-freestanding -Doptimize=ReleaseSmall
```

**WASM Interface (JavaScript):**
```zig
// Exports for JavaScript
export fn train_step(tokens_ptr: [*]const u32, targets_ptr: [*]const u32, len: usize) f32 {
    const tokens = tokens_ptr[0..len];
    const targets = targets_ptr[0..len];
    
    const logits = model.forward(tokens) catch return -1.0;
    defer allocator.free(logits);
    
    const loss = crossEntropyLoss(logits, targets, len, model.config.vocab_size);
    
    const grad = crossEntropyGrad(allocator, logits, targets, len, model.config.vocab_size) catch return -1.0;
    defer allocator.free(grad);
    
    model.backward(grad) catch return -1.0;
    
    optimizer.step(&model, learning_rate, weight_decay) catch return -1.0;
    model.zeroGrad();
    
    return loss;
}

export fn generate(prompt_ptr: [*]const u32, prompt_len: usize, max_tokens: usize) [*]u32 {
    // Autoregressive generation
    // Returns pointer to generated tokens (caller must free)
    _ = prompt_ptr;
    _ = prompt_len;
    _ = max_tokens;
    unreachable; // TODO: implement
}

export fn save_checkpoint() [*]u8 {
    // Serialize model to bytes
    // Returns pointer to checkpoint data
    unreachable; // TODO: implement
}

export fn load_checkpoint(data_ptr: [*]const u8, len: usize) bool {
    // Deserialize model from bytes
    _ = data_ptr;
    _ = len;
    unreachable; // TODO: implement
}
```

## Development Workflow

### mise Configuration (0.15.1)

**.mise.toml:**
```toml
[tools]
zig = "0.15.1"
mdbook = "latest"

[tasks]
test = "zig build test"
test-watch = "zig build test --watch"
docs = "mdbook build docs"
docs-serve = "mdbook serve docs"
train = "zig build run -- train --config config.json"
bench = "zig build run -- benchmark"
fmt = "zig fmt src/ tests/"
clean = "rm -rf zig-cache zig-out .zig-cache"

[env]
ZIG_LOCAL_CACHE_DIR = ".zig-cache"
```

**Setup:**
```bash
# Install mise (if not already)
curl https://mise.run | sh

# Install tools and activate environment
mise install
mise trust

# Run tasks
mise run test
mise run docs-serve
mise run train
```

### build.zig (0.15.1 CRITICAL UPDATE)

**CORRECT 0.15.1 build.zig:**
```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Create root module (NEW in 0.14, REQUIRED in 0.15)
    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Library
    const lib = b.addStaticLibrary(.{
        .name = "zeptochat",
        .root_module = root_module,
    });
    b.installArtifact(lib);
    
    // Executable
    const exe = b.addExecutable(.{
        .name = "zeptochat",
        .root_module = root_module,
    });
    b.installArtifact(exe);
    
    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
    
    // Tests - use separate test root module
    const test_root_module = b.createModule(.{
        .root_source_file = b.path("tests/all_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const main_tests = b.addTest(.{
        .root_module = test_root_module,
    });
    
    const run_tests = b.addRunArtifact(main_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);
    
    // Benchmark - separate module
    const bench_root_module = b.createModule(.{
        .root_source_file = b.path("examples/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    
    const bench = b.addExecutable(.{
        .name = "benchmark",
        .root_module = bench_root_module,
    });
    
    const run_bench = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
    
    // WASM target
    if (target.result.cpu.arch == .wasm32) {
        exe.entry = .disabled;
        exe.rdynamic = true;
    }
}
```

### Testing Strategy (0.15.1)

**Test Aggregator (tests/all_tests.zig):**
```zig
test {
    _ = @import("tokenizer_test.zig");
    _ = @import("transformer_test.zig");
    _ = @import("simd_test.zig");
    _ = @import("training_test.zig");
    _ = @import("gradient_test.zig");
}
```

**Gradient Testing (tests/gradient_test.zig):**
```zig
const std = @import("std");
const testing = std.testing;
const Transformer = @import("../src/transformer.zig").Transformer;

test "gradient check - attention" {
    const allocator = testing.allocator;
    
    // Small model for testing
    const config = .{
        .vocab_size = 100,
        .context_length = 16,
        .d_model = 64,
        .n_heads = 4,
        .n_layers = 2,
        .dropout = 0.0,
    };
    
    var model = try Transformer.init(allocator, config);
    defer model.deinit();
    
    // Create test input
    const tokens = [_]u32{1, 2, 3, 4};
    
    // Numerical gradient
    const eps = 1e-4;
    const param = &model.layers[0].attn.w_q[0];
    const original = param.*;
    
    param.* = original + eps;
    const logits_plus = try model.forward(&tokens);
    defer allocator.free(logits_plus);
    const loss_plus = sum(logits_plus);
    
    param.* = original - eps;
    const logits_minus = try model.forward(&tokens);
    defer allocator.free(logits_minus);
    const loss_minus = sum(logits_minus);
    
    const numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
    
    // Analytical gradient
    param.* = original;
    const logits = try model.forward(&tokens);
    defer allocator.free(logits);
    
    // Create dummy gradient (all ones for simplicity)
    const grad = try allocator.alloc(f32, logits.len);
    defer allocator.free(grad);
    @memset(grad, 1.0);
    
    try model.backward(grad);
    const analytical_grad = param.grad;
    
    // Should match within tolerance
    const diff = @abs(numerical_grad - analytical_grad);
    const relative_error = diff / (@abs(numerical_grad) + @abs(analytical_grad) + 1e-8);
    
    try testing.expect(relative_error < 1e-3);
}

fn sum(values: []const f32) f32 {
    var total: f32 = 0;
    for (values) |v| total += v;
    return total;
}
```

**Overfit Test (tests/training_test.zig):**
```zig
const std = @import("std");
const testing = std.testing;

test "overfit single batch" {
    const allocator = testing.allocator;
    
    const small_config = .{
        .vocab_size = 50,
        .context_length = 16,
        .d_model = 64,
        .n_heads = 4,
        .n_layers = 2,
        .dropout = 0.0,
    };
    
    var model = try Transformer.init(allocator, small_config);
    defer model.deinit();
    
    var optimizer = try AdamW.init(allocator, model.numParameters());
    defer optimizer.deinit();
    
    // Single batch
    const tokens = [_]u32{1, 2, 3, 4, 5};
    const targets = [_]u32{2, 3, 4, 5, 6};
    
    var initial_loss: f32 = undefined;
    
    // Train for 100 steps
    for (0..100) |step| {
        const logits = try model.forward(&tokens);
        defer allocator.free(logits);
        
        const loss = crossEntropyLoss(logits, &targets, tokens.len, small_config.vocab_size);
        
        if (step == 0) initial_loss = loss;
        
        const grad = try crossEntropyGrad(allocator, logits, &targets, tokens.len, small_config.vocab_size);
        defer allocator.free(grad);
        
        try model.backward(grad);
        try optimizer.step(&model, 0.001, 0.01);
        model.zeroGrad();
    }
    
    // Final loss should be much lower
    const logits = try model.forward(&tokens);
    defer allocator.free(logits);
    const final_loss = crossEntropyLoss(logits, &targets, tokens.len, small_config.vocab_size);
    
    try testing.expect(final_loss < initial_loss * 0.1);
}
```

## Model Specifications

### Tiny Model (CPU Training Test)

**Purpose:** Fastest training, verify implementation correctness

```zig
pub const TinyConfig = ModelConfig{
    .vocab_size = 256,           // Byte-level
    .context_length = 128,       // Short context
    .d_model = 128,              // Small dimension
    .n_heads = 4,                // Few heads
    .n_layers = 4,               // Few layers
    .dropout = 0.0,              // No dropout for overfitting test
};
```

**Estimated Parameters:** ~1M parameters
**Training Target:** Overfit on 1KB of text in < 1 minute on modern CPU
**Use Case:** Development, testing, gradient checking

### Small Model (Consumer CPU)

**Purpose:** Realistic training on consumer hardware

```zig
pub const SmallConfig = ModelConfig{
    .vocab_size = 50257,         // GPT-2 vocab
    .context_length = 256,       // Reasonable context
    .d_model = 256,              // Small but useful
    .n_heads = 8,
    .n_layers = 6,
    .dropout = 0.1,
};
```

**Estimated Parameters:** ~10M parameters
**Training Target:** Few hours on 8-core CPU, converge on TinyStories
**Use Case:** Proof of concept, educational

## CLI Interface (0.15.1)

**Command Structure:**
```bash
# Training
zeptochat train \
    --config configs/small.json \
    --data tinystories.tokens \
    --output checkpoints/ \
    --epochs 10

# Resume from checkpoint
zeptochat train \
    --config configs/small.json \
    --data tinystories.tokens \
    --resume checkpoints/step_1000.ckpt

# Evaluation
zeptochat eval \
    --checkpoint checkpoints/final.ckpt \
    --data tinystories_val.tokens

# Generation
zeptochat generate \
    --checkpoint checkpoints/final.ckpt \
    --prompt "Once upon a time" \
    --max-tokens 100 \
    --temperature 0.8

# Benchmark
zeptochat benchmark \
    --config configs/small.json \
    --operations matmul,attention,mlp
```

## Implementation Checklist

Use this as a roadmap for the coding agent:

### Setup
- [ ] Initialize git repo
- [ ] Create directory structure
- [ ] Setup build.zig (0.15.1 with root_module)
- [ ] Setup .mise.toml with Zig 0.15.1
- [ ] Create README.md
- [ ] Setup mdBook docs structure

### Phase 1: CPU Baseline (0.15.1 Compatible)
- [ ] Tokenizer
  - [ ] BPE encode
  - [ ] BPE decode
  - [ ] Vocab/merges loading
  - [ ] Tests (round-trip, known examples)
  
- [ ] Transformer
  - [ ] Embeddings (token + position)
  - [ ] Multi-head attention
  - [ ] MLP with GELU
  - [ ] Layer norm
  - [ ] Forward pass
  - [ ] Tests (shape checking, reference comparison)
  
- [ ] Backpropagation
  - [ ] Manual gradient computation
  - [ ] Backward pass implementation
  - [ ] Tests (gradient checking)
  
- [ ] Optimizer (AdamW)
  - [ ] Implementation with proper weight decay
  - [ ] Parameter updates
  - [ ] Tests (convergence on quadratic)
  
- [ ] Training Loop
  - [ ] Data loading (0.15.1 Reader API)
  - [ ] Batch iteration
  - [ ] Forward/backward/update loop
  - [ ] Loss calculation
  - [ ] Logging (0.15.1 Writer API with buffers)
  - [ ] Tests (overfit single batch)

### Phase 2: SIMD
- [ ] Benchmark harness
- [ ] Scalar implementations (baseline)
- [ ] AVX2 implementations
- [ ] NEON implementations (ARM)
- [ ] Comptime dispatch (0.15.1 style)
- [ ] Tests (SIMD matches scalar)
- [ ] Performance measurements

### Phase 3: Multi-threading
- [ ] Thread pool using std.Thread
- [ ] Parallel matmul
- [ ] Thread-safe gradient accumulation
- [ ] Tests (multi-threaded matches single)
- [ ] Scaling benchmarks

### Phase 4: Full Pipeline
- [ ] Dataset preparation scripts
- [ ] Efficient data loading
- [ ] Checkpoint save/load
- [ ] Evaluation loop
- [ ] Logging with 0.15.1 Writer API
- [ ] Generation (sampling)
- [ ] Train Small model
- [ ] Validate generation quality

### Phase 5: WASM
- [ ] WASM build configuration
- [ ] JavaScript bindings
- [ ] Memory management
- [ ] Web UI (HTML/CSS/JS)
- [ ] IndexedDB integration
- [ ] Web Workers
- [ ] Deployment
- [ ] Browser testing

### Phase 6: Documentation
- [ ] Complete mdBook content
- [ ] Code comments
- [ ] README examples
- [ ] CONTRIBUTING guide
- [ ] Performance documentation
- [ ] Blog post

## Critical 0.15.1 Migration Patterns

### DO NOT USE (Old 0.14 patterns):
```zig
// WRONG - removed in 0.15
pub usingnamespace SomeMixin;

// WRONG - removed in 0.15
const stdout = std.io.getStdOut().writer();

// WRONG - removed in 0.15
const exe = b.addExecutable(.{
    .root_source_file = b.path("src/main.zig"),
});

// WRONG - removed in 0.15
pub fn format(
    this: @This(),
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void
```

### DO USE (Correct 0.15.1 patterns):
```zig
// CORRECT - explicit declarations
pub const foo = if (have_feature) realFoo else dummyFoo;

// CORRECT - buffer-based I/O
var stdout_buffer: [4096]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout: *std.Io.Writer = &stdout_writer.interface;
try stdout.flush(); // DON'T FORGET

// CORRECT - root_module field
const root_module = b.createModule(.{
    .root_source_file = b.path("src/main.zig"),
});
const exe = b.addExecutable(.{
    .name = "foo",
    .root_module = root_module,
});

// CORRECT - simplified format signature
pub fn format(this: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void

// CORRECT - use {f} for format methods
try writer.print("{f}", .{my_custom_type});

// CORRECT - ArrayListUnmanaged is default now
var list = std.ArrayListUnmanaged(i32){};
try list.append(allocator, 42);

// CORRECT - inline assembly clobbers
: .{ .rcx = true, .r11 = true }
```

## References

**Zig 0.15.1 Specific:**
- Release Notes: https://ziglang.org/download/0.15.1/release-notes.html
- Documentation: https://ziglang.org/documentation/0.15.1/

**Inspirations:**
- nanochat by Andrej Karpathy - https://github.com/karpathy/nanochat
- nanoGPT by Andrej Karpathy - https://github.com/karpathy/nanoGPT
- TigerStyle - https://tigerstyle.dev

**Technical Resources:**
- Attention Is All You Need - https://arxiv.org/abs/1706.03762
- The Illustrated Transformer - https://jalammar.github.io/illustrated-transformer/
