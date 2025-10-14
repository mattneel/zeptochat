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
    grad_weight: []f32,
    grad_bias: []f32,
    cached_input: ?[]f32 = null,
    cached_mean: f32 = 0,
    cached_std: f32 = 0,
    allocator: std.mem.Allocator,
    eps: f32 = 1e-5,

    pub fn init(allocator: std.mem.Allocator, d_model: usize) !LayerNorm {
        const weight = try allocator.alloc(f32, d_model);
        errdefer allocator.free(weight);

        const bias = try allocator.alloc(f32, d_model);
        errdefer allocator.free(bias);

        const grad_weight = try allocator.alloc(f32, d_model);
        errdefer allocator.free(grad_weight);

        const grad_bias = try allocator.alloc(f32, d_model);
        errdefer allocator.free(grad_bias);

        @memset(weight, 1.0);
        @memset(bias, 0.0);
        @memset(grad_weight, 0.0);
        @memset(grad_bias, 0.0);

        return LayerNorm{
            .weight = weight,
            .bias = bias,
            .grad_weight = grad_weight,
            .grad_bias = grad_bias,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LayerNorm) void {
        self.allocator.free(self.weight);
        self.allocator.free(self.bias);
        self.allocator.free(self.grad_weight);
        self.allocator.free(self.grad_bias);
        if (self.cached_input) |buf| {
            self.allocator.free(buf);
        }
    }

    pub fn forward(self: *LayerNorm, x: []f32) !void {
        const n = x.len;
        const n_f32: f32 = @floatFromInt(n);

        if (self.cached_input) |buf| {
            self.allocator.free(buf);
        }
        self.cached_input = try self.allocator.alloc(f32, n);
        @memcpy(self.cached_input.?, x);

        var sum: f32 = 0;
        for (x) |val| sum += val;
        const mean = sum / n_f32;
        self.cached_mean = mean;

        var var_sum: f32 = 0;
        for (x) |val| {
            const diff = val - mean;
            var_sum += diff * diff;
        }
        const variance = var_sum / n_f32;
        const std_dev = @sqrt(variance + self.eps);
        self.cached_std = std_dev;
        const inv_std = 1.0 / std_dev;

        for (x, 0..) |*val, i| {
            val.* = (val.* - mean) * inv_std * self.weight[i] + self.bias[i];
        }
    }

    pub fn backward(self: *LayerNorm, allocator: std.mem.Allocator, grad_output: []const f32) ![]f32 {
        const input = self.cached_input orelse return error.ForwardNotCalled;
        const n = grad_output.len;
        const n_f32: f32 = @floatFromInt(n);

        var x_norm = try allocator.alloc(f32, n);
        defer allocator.free(x_norm);

        for (input, 0..) |val, i| {
            x_norm[i] = (val - self.cached_mean) / self.cached_std;
        }

        for (grad_output, 0..) |grad, i| {
            self.grad_weight[i] += grad * x_norm[i];
            self.grad_bias[i] += grad;
        }

        var sum_scaled: f32 = 0;
        var sum_scaled_norm: f32 = 0;

        for (grad_output, 0..) |grad, i| {
            const scaled = grad * self.weight[i];
            sum_scaled += scaled;
            sum_scaled_norm += scaled * x_norm[i];
        }

        var grad_input = try allocator.alloc(f32, n);

        for (grad_output, 0..) |grad, i| {
            const scaled = grad * self.weight[i];
            grad_input[i] = (scaled - sum_scaled / n_f32 - x_norm[i] * sum_scaled_norm / n_f32) / self.cached_std;
        }

        return grad_input;
    }

    pub fn zeroGrad(self: *LayerNorm) void {
        @memset(self.grad_weight, 0.0);
        @memset(self.grad_bias, 0.0);
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

pub const MLP = struct {
    d_model: usize,
    d_ff: usize,
    w1: []f32,
    b1: []f32,
    w2: []f32,
    b2: []f32,
    grad_w1: []f32,
    grad_b1: []f32,
    grad_w2: []f32,
    grad_b2: []f32,
    cached_input: ?[]f32 = null,
    cached_pre: ?[]f32 = null,
    cached_hidden: ?[]f32 = null,
    seq_len: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, d_model: usize) !MLP {
        const d_ff = d_model * 4;

        const w1 = try allocator.alloc(f32, d_model * d_ff);
        errdefer allocator.free(w1);

        const b1 = try allocator.alloc(f32, d_ff);
        errdefer allocator.free(b1);

        const w2 = try allocator.alloc(f32, d_ff * d_model);
        errdefer allocator.free(w2);

        const b2 = try allocator.alloc(f32, d_model);
        errdefer allocator.free(b2);

        const grad_w1 = try allocator.alloc(f32, d_model * d_ff);
        errdefer allocator.free(grad_w1);

        const grad_b1 = try allocator.alloc(f32, d_ff);
        errdefer allocator.free(grad_b1);

        const grad_w2 = try allocator.alloc(f32, d_ff * d_model);
        errdefer allocator.free(grad_w2);

        const grad_b2 = try allocator.alloc(f32, d_model);
        errdefer allocator.free(grad_b2);

        var prng = std.Random.DefaultPrng.init(2024);
        const random = prng.random();
        const scale1 = @sqrt(1.0 / @as(f32, @floatFromInt(d_model)));
        const scale2 = @sqrt(1.0 / @as(f32, @floatFromInt(d_ff)));

        for (w1) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale1;
        for (w2) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale2;

        @memset(b1, 0.0);
        @memset(b2, 0.0);
        @memset(grad_w1, 0.0);
        @memset(grad_b1, 0.0);
        @memset(grad_w2, 0.0);
        @memset(grad_b2, 0.0);

        return MLP{
            .d_model = d_model,
            .d_ff = d_ff,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
            .grad_w1 = grad_w1,
            .grad_b1 = grad_b1,
            .grad_w2 = grad_w2,
            .grad_b2 = grad_b2,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MLP) void {
        self.allocator.free(self.w1);
        self.allocator.free(self.b1);
        self.allocator.free(self.w2);
        self.allocator.free(self.b2);
        self.allocator.free(self.grad_w1);
        self.allocator.free(self.grad_b1);
        self.allocator.free(self.grad_w2);
        self.allocator.free(self.grad_b2);
        if (self.cached_input) |buf| self.allocator.free(buf);
        if (self.cached_pre) |buf| self.allocator.free(buf);
        if (self.cached_hidden) |buf| self.allocator.free(buf);
    }

    pub fn forward(
        self: *MLP,
        allocator: std.mem.Allocator,
        x: []const f32,
        seq_len: usize,
    ) ![]f32 {
        self.seq_len = seq_len;

        if (self.cached_input) |buf| allocator.free(buf);
        self.cached_input = try allocator.alloc(f32, x.len);
        @memcpy(self.cached_input.?, x);

        var pre = try matmul(allocator, x, self.w1, seq_len, self.d_model, self.d_ff);
        defer allocator.free(pre);

        for (0..seq_len) |i| {
            const offset = i * self.d_ff;
            for (0..self.d_ff) |j| {
                pre[offset + j] += self.b1[j];
            }
        }

        if (self.cached_pre) |buf| allocator.free(buf);
        self.cached_pre = try allocator.alloc(f32, pre.len);
        @memcpy(self.cached_pre.?, pre);

        gelu(pre);

        if (self.cached_hidden) |buf| allocator.free(buf);
        self.cached_hidden = try allocator.alloc(f32, pre.len);
        @memcpy(self.cached_hidden.?, pre);

        var output = try matmul(allocator, pre, self.w2, seq_len, self.d_ff, self.d_model);

        for (0..seq_len) |i| {
            const offset = i * self.d_model;
            for (0..self.d_model) |j| {
                output[offset + j] += self.b2[j];
            }
        }

        return output;
    }

    pub fn backward(
        self: *MLP,
        allocator: std.mem.Allocator,
        grad_output: []const f32,
    ) ![]f32 {
        const seq_len = self.seq_len;
        const input = self.cached_input orelse return error.ForwardNotCalled;
        const pre = self.cached_pre orelse return error.ForwardNotCalled;
        const hidden = self.cached_hidden orelse return error.ForwardNotCalled;

        for (0..seq_len) |i| {
            const offset = i * self.d_model;
            for (0..self.d_model) |j| {
                self.grad_b2[j] += grad_output[offset + j];
            }
        }

        for (0..seq_len) |i| {
            for (0..self.d_ff) |j| {
                for (0..self.d_model) |k| {
                    self.grad_w2[j * self.d_model + k] +=
                        hidden[i * self.d_ff + j] * grad_output[i * self.d_model + k];
                }
            }
        }

        var grad_hidden = try allocator.alloc(f32, seq_len * self.d_ff);
        defer allocator.free(grad_hidden);

        for (0..seq_len) |i| {
            for (0..self.d_ff) |j| {
                var sum: f32 = 0;
                for (0..self.d_model) |k| {
                    sum += grad_output[i * self.d_model + k] * self.w2[j * self.d_model + k];
                }
                grad_hidden[i * self.d_ff + j] = sum;
            }
        }

        geluBackward(grad_hidden, pre);

        for (0..seq_len) |i| {
            const offset = i * self.d_ff;
            for (0..self.d_ff) |j| {
                self.grad_b1[j] += grad_hidden[offset + j];
            }
        }

        for (0..seq_len) |i| {
            for (0..self.d_model) |j| {
                for (0..self.d_ff) |k| {
                    self.grad_w1[j * self.d_ff + k] +=
                        input[i * self.d_model + j] * grad_hidden[i * self.d_ff + k];
                }
            }
        }

        var grad_input = try allocator.alloc(f32, seq_len * self.d_model);

        for (0..seq_len) |i| {
            for (0..self.d_model) |j| {
                var sum: f32 = 0;
                for (0..self.d_ff) |k| {
                    sum += grad_hidden[i * self.d_ff + k] * self.w1[j * self.d_ff + k];
                }
                grad_input[i * self.d_model + j] = sum;
            }
        }

        return grad_input;
    }

    pub fn zeroGrad(self: *MLP) void {
        @memset(self.grad_w1, 0.0);
        @memset(self.grad_b1, 0.0);
        @memset(self.grad_w2, 0.0);
        @memset(self.grad_b2, 0.0);
    }
};

pub const TransformerBlock = struct {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    mlp: MLP,

    pub fn init(allocator: std.mem.Allocator, d_model: usize, n_heads: usize) !TransformerBlock {
        var ln1 = try LayerNorm.init(allocator, d_model);
        errdefer ln1.deinit();

        var attn = try MultiHeadAttention.init(allocator, d_model, n_heads);
        errdefer attn.deinit();

        var ln2 = try LayerNorm.init(allocator, d_model);
        errdefer ln2.deinit();

        var mlp = try MLP.init(allocator, d_model);
        errdefer mlp.deinit();

        return TransformerBlock{
            .ln1 = ln1,
            .attn = attn,
            .ln2 = ln2,
            .mlp = mlp,
        };
    }

    pub fn deinit(self: *TransformerBlock) void {
        self.ln1.deinit();
        self.attn.deinit();
        self.ln2.deinit();
        self.mlp.deinit();
    }

    pub fn forward(
        self: *TransformerBlock,
        allocator: std.mem.Allocator,
        x: []const f32,
        seq_len: usize,
        d_model: usize,
    ) ![]f32 {
        var norm1 = try allocator.alloc(f32, x.len);
        defer allocator.free(norm1);
        @memcpy(norm1, x);

        for (0..seq_len) |i| {
            const offset = i * d_model;
            try self.ln1.forward(norm1[offset..][0..d_model]);
        }

        const attn_out = try self.attn.forward(allocator, norm1, seq_len);
        defer allocator.free(attn_out);

        var residual = try allocator.alloc(f32, x.len);
        defer allocator.free(residual);
        for (0..x.len) |i| {
            residual[i] = x[i] + attn_out[i];
        }

        var norm2 = try allocator.alloc(f32, residual.len);
        defer allocator.free(norm2);
        @memcpy(norm2, residual);

        for (0..seq_len) |i| {
            const offset = i * d_model;
            try self.ln2.forward(norm2[offset..][0..d_model]);
        }

        const mlp_out = try self.mlp.forward(allocator, norm2, seq_len);
        defer allocator.free(mlp_out);

        const output = try allocator.alloc(f32, residual.len);
        for (0..residual.len) |i| {
            output[i] = residual[i] + mlp_out[i];
        }

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

pub fn gelu(x: []f32) void {
    const sqrt_2_over_pi = @sqrt(2.0 / std.math.pi);

    for (x) |*val| {
        const x_val = val.*;
        const x_cubed = x_val * x_val * x_val;
        const inner = sqrt_2_over_pi * (x_val + 0.044715 * x_cubed);
        const cdf = 0.5 * (1.0 + std.math.tanh(inner));
        val.* = x_val * cdf;
    }
}

fn geluBackward(grad: []f32, x: []const f32) void {
    const sqrt_2_over_pi = @sqrt(2.0 / std.math.pi);

    for (grad, x) |*g, x_val| {
        const x_cubed = x_val * x_val * x_val;
        const inner = sqrt_2_over_pi * (x_val + 0.044715 * x_cubed);
        const tanh_inner = std.math.tanh(inner);
        const cdf = 0.5 * (1.0 + tanh_inner);

        const sech_sq = 1.0 - tanh_inner * tanh_inner;
        const d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x_val * x_val);
        const gelu_grad = cdf + x_val * 0.5 * sech_sq * d_inner;

        g.* *= gelu_grad;
    }
}

pub const Transformer = struct {
    config: ModelConfig,
    allocator: std.mem.Allocator,
    token_embeddings: []f32,
    position_embeddings: []f32,
    layers: []TransformerBlock,
    ln_final: LayerNorm,
    lm_head: []f32,

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

        const layers = try allocator.alloc(TransformerBlock, config.n_layers);
        errdefer allocator.free(layers);

        for (0..config.n_layers) |i| {
            layers[i] = try TransformerBlock.init(allocator, config.d_model, config.n_heads);
            errdefer {
                for (0..i) |j| layers[j].deinit();
            }
        }

        var ln_final = try LayerNorm.init(allocator, config.d_model);
        errdefer ln_final.deinit();

        const lm_head = try allocator.alloc(f32, config.d_model * config.vocab_size);
        errdefer allocator.free(lm_head);

        const head_scale = @sqrt(1.0 / @as(f32, @floatFromInt(config.d_model)));
        for (lm_head) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * head_scale;
        }

        return Transformer{
            .config = config,
            .allocator = allocator,
            .token_embeddings = token_embeddings,
            .position_embeddings = position_embeddings,
            .layers = layers,
            .ln_final = ln_final,
            .lm_head = lm_head,
        };
    }

    pub fn deinit(self: *Transformer) void {
        self.allocator.free(self.token_embeddings);
        self.allocator.free(self.position_embeddings);
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.ln_final.deinit();
        self.allocator.free(self.lm_head);
    }

    pub fn forward(self: *Transformer, tokens: []const u32) ![]f32 {
        const seq_len = tokens.len;
        const d_model = self.config.d_model;
        const vocab_size = self.config.vocab_size;

        var hidden = try self.allocator.alloc(f32, seq_len * d_model);
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

        for (self.layers) |*layer| {
            const new_hidden = try layer.forward(self.allocator, hidden, seq_len, d_model);
            self.allocator.free(hidden);
            hidden = new_hidden;
        }

        for (0..seq_len) |i| {
            const offset = i * d_model;
            try self.ln_final.forward(hidden[offset..][0..d_model]);
        }

        const logits = try matmul(self.allocator, hidden, self.lm_head, seq_len, d_model, vocab_size);
        self.allocator.free(hidden);
        return logits;
    }
};
