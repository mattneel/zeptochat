const std = @import("std");

pub const ModelConfig = struct {
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    dropout: f32,
};

/// Minimal configuration for rapid iteration and unit tests.
pub const TinyConfig = ModelConfig{
    .vocab_size = 256,
    .context_length = 32,
    .d_model = 64,
    .n_heads = 2,
    .n_layers = 2,
    .dropout = 0.0,
};

pub const LayerNorm = struct {
    weight: []f32,
    bias: []f32,
    allocator: std.mem.Allocator,
    eps: f32 = 1e-5,

    pub fn init(allocator: std.mem.Allocator, d_model: usize) !LayerNorm {
        const weight = try allocator.alloc(f32, d_model);
        errdefer allocator.free(weight);

        const bias = try allocator.alloc(f32, d_model);
        errdefer allocator.free(bias);

        @memset(weight, 1.0);
        @memset(bias, 0.0);

        return LayerNorm{
            .weight = weight,
            .bias = bias,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LayerNorm) void {
        self.allocator.free(self.weight);
        self.allocator.free(self.bias);
    }

    pub fn forward(self: *const LayerNorm, x: []f32) void {
        const n = x.len;
        const n_f32: f32 = @floatFromInt(n);

        var sum: f32 = 0;
        for (x) |val| sum += val;
        const mean = sum / n_f32;

        var var_sum: f32 = 0;
        for (x) |val| {
            const diff = val - mean;
            var_sum += diff * diff;
        }
        const variance = var_sum / n_f32;
        const inv_std = 1.0 / @sqrt(variance + self.eps);

        for (x, 0..) |*val, i| {
            val.* = (val.* - mean) * inv_std * self.weight[i] + self.bias[i];
        }
    }
};

pub const MultiHeadAttention = struct {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    w_q: []f32,
    w_k: []f32,
    w_v: []f32,
    w_o: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, d_model: usize, n_heads: usize) !MultiHeadAttention {
        std.debug.assert(d_model % n_heads == 0);

        const head_dim = d_model / n_heads;
        const d_squared = d_model * d_model;

        const w_q = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(w_q);

        const w_k = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(w_k);

        const w_v = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(w_v);

        const w_o = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(w_o);

        var prng = std.Random.DefaultPrng.init(1337);
        const random = prng.random();
        const scale = @sqrt(1.0 / @as(f32, @floatFromInt(d_model)));

        for (w_q) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (w_k) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (w_v) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (w_o) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;

        return MultiHeadAttention{
            .d_model = d_model,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MultiHeadAttention) void {
        self.allocator.free(self.w_q);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_v);
        self.allocator.free(self.w_o);
    }

    pub fn forward(
        self: *const MultiHeadAttention,
        allocator: std.mem.Allocator,
        x: []const f32,
        seq_len: usize,
    ) ![]f32 {
        const d_model = self.d_model;
        const n_heads = self.n_heads;
        const head_dim = self.head_dim;

        const q = try matmul(allocator, x, self.w_q, seq_len, d_model, d_model);
        defer allocator.free(q);

        const k = try matmul(allocator, x, self.w_k, seq_len, d_model, d_model);
        defer allocator.free(k);

        const v = try matmul(allocator, x, self.w_v, seq_len, d_model, d_model);
        defer allocator.free(v);

        var head_outputs = try allocator.alloc([]f32, n_heads);
        defer {
            for (head_outputs) |output| allocator.free(output);
            allocator.free(head_outputs);
        }

        for (0..n_heads) |h| {
            const head_offset = h * head_dim;

            var q_head = try allocator.alloc(f32, seq_len * head_dim);
            defer allocator.free(q_head);
            var k_head = try allocator.alloc(f32, seq_len * head_dim);
            defer allocator.free(k_head);
            var v_head = try allocator.alloc(f32, seq_len * head_dim);
            defer allocator.free(v_head);

            for (0..seq_len) |i| {
                const src_offset = i * d_model + head_offset;
                const dst_offset = i * head_dim;
                @memcpy(q_head[dst_offset..][0..head_dim], q[src_offset..][0..head_dim]);
                @memcpy(k_head[dst_offset..][0..head_dim], k[src_offset..][0..head_dim]);
                @memcpy(v_head[dst_offset..][0..head_dim], v[src_offset..][0..head_dim]);
            }

            head_outputs[h] = try scaledDotProductAttention(
                allocator,
                q_head,
                k_head,
                v_head,
                seq_len,
                head_dim,
            );
        }

        var concat = try allocator.alloc(f32, seq_len * d_model);
        defer allocator.free(concat);

        for (0..n_heads) |h| {
            const head_offset = h * head_dim;
            for (0..seq_len) |i| {
                const dst_offset = i * d_model + head_offset;
                const src_offset = i * head_dim;
                @memcpy(concat[dst_offset..][0..head_dim], head_outputs[h][src_offset..][0..head_dim]);
            }
        }

        const output = try matmul(allocator, concat, self.w_o, seq_len, d_model, d_model);
        return output;
    }
};

fn scaledDotProductAttention(
    allocator: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    seq_len: usize,
    head_dim: usize,
) ![]f32 {
    const scores = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(scores);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            var dot: f32 = 0;
            for (0..head_dim) |d| {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            if (j > i) {
                scores[i * seq_len + j] = -std.math.inf(f32);
            }
        }
    }

    for (0..seq_len) |i| {
        softmax(scores[i * seq_len ..][0..seq_len]);
    }

    const output = try allocator.alloc(f32, seq_len * head_dim);

    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            var sum: f32 = 0;
            for (0..seq_len) |j| {
                sum += scores[i * seq_len + j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }

    return output;
}

fn softmax(x: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (x) |val| max = @max(max, val);

    var sum: f32 = 0;
    for (x) |*val| {
        val.* = @exp(val.* - max);
        sum += val.*;
    }

    for (x) |*val| val.* /= sum;
}

fn matmul(
    allocator: std.mem.Allocator,
    a: []const f32,
    b: []const f32,
    M: usize,
    K: usize,
    N: usize,
) ![]f32 {
    const c = try allocator.alloc(f32, M * N);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }

    return c;
}

pub const Transformer = struct {
    config: ModelConfig,
    allocator: std.mem.Allocator,
    token_embeddings: []f32,
    position_embeddings: []f32,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !Transformer {
        const token_embeddings = try allocator.alloc(f32, config.vocab_size * config.d_model);
        errdefer allocator.free(token_embeddings);

        const position_embeddings = try allocator.alloc(f32, config.context_length * config.d_model);
        errdefer allocator.free(position_embeddings);

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const scale = @sqrt(1.0 / @as(f32, @floatFromInt(config.d_model)));

        for (token_embeddings) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }

        for (position_embeddings) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }

        return Transformer{
            .config = config,
            .allocator = allocator,
            .token_embeddings = token_embeddings,
            .position_embeddings = position_embeddings,
        };
    }

    pub fn deinit(self: *Transformer) void {
        self.allocator.free(self.token_embeddings);
        self.allocator.free(self.position_embeddings);
    }

    pub fn forward(self: *Transformer, tokens: []const u32) ![]f32 {
        const batch_size = tokens.len;
        const d_model = self.config.d_model;

        var hidden = try self.allocator.alloc(f32, batch_size * d_model);
        errdefer self.allocator.free(hidden);

        for (tokens, 0..) |token, pos| {
            const token_offset = token * d_model;
            const pos_offset = pos * d_model;
            const hidden_offset = pos * d_model;

            for (0..d_model) |i| {
                hidden[hidden_offset + i] =
                    self.token_embeddings[token_offset + i] +
                    self.position_embeddings[pos_offset + i];
            }
        }

        return hidden;
    }
};
