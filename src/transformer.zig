const std = @import("std");
const parallel = @import("parallel");

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

var global_thread_pool: ?*parallel.ThreadPool = null;

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
    grad_w_q: []f32,
    grad_w_k: []f32,
    grad_w_v: []f32,
    grad_w_o: []f32,
    cached_input: ?[]f32 = null,
    cached_q: ?[]f32 = null,
    cached_k: ?[]f32 = null,
    cached_v: ?[]f32 = null,
    cached_attn_output: ?[]f32 = null,
    cached_attn_weights: ?[]f32 = null,
    seq_len: usize = 0,
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

        const grad_w_q = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(grad_w_q);

        const grad_w_k = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(grad_w_k);

        const grad_w_v = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(grad_w_v);

        const grad_w_o = try allocator.alloc(f32, d_squared);
        errdefer allocator.free(grad_w_o);

        var prng = std.Random.DefaultPrng.init(1337);
        const random = prng.random();
        const scale = @sqrt(1.0 / @as(f32, @floatFromInt(d_model)));

        for (w_q) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (w_k) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (w_v) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (w_o) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;

        @memset(grad_w_q, 0.0);
        @memset(grad_w_k, 0.0);
        @memset(grad_w_v, 0.0);
        @memset(grad_w_o, 0.0);

        return MultiHeadAttention{
            .d_model = d_model,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .grad_w_q = grad_w_q,
            .grad_w_k = grad_w_k,
            .grad_w_v = grad_w_v,
            .grad_w_o = grad_w_o,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MultiHeadAttention) void {
        self.allocator.free(self.w_q);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_v);
        self.allocator.free(self.w_o);
        self.allocator.free(self.grad_w_q);
        self.allocator.free(self.grad_w_k);
        self.allocator.free(self.grad_w_v);
        self.allocator.free(self.grad_w_o);

        if (self.cached_input) |buf| self.allocator.free(buf);
        if (self.cached_q) |buf| self.allocator.free(buf);
        if (self.cached_k) |buf| self.allocator.free(buf);
        if (self.cached_v) |buf| self.allocator.free(buf);
        if (self.cached_attn_output) |buf| self.allocator.free(buf);
        if (self.cached_attn_weights) |buf| self.allocator.free(buf);
    }

    pub fn forward(
        self: *MultiHeadAttention,
        allocator: std.mem.Allocator,
        x: []const f32,
        seq_len: usize,
    ) ![]f32 {
        self.seq_len = seq_len;

        if (self.cached_input) |buf| allocator.free(buf);
        self.cached_input = try allocator.alloc(f32, x.len);
        @memcpy(self.cached_input.?, x);

        if (self.cached_q) |buf| allocator.free(buf);
        const q = try matmul(allocator, x, self.w_q, seq_len, self.d_model, self.d_model);
        self.cached_q = q;
        errdefer {
            if (self.cached_q) |buf| {
                allocator.free(buf);
                self.cached_q = null;
            }
        }

        if (self.cached_k) |buf| allocator.free(buf);
        const k = try matmul(allocator, x, self.w_k, seq_len, self.d_model, self.d_model);
        self.cached_k = k;
        errdefer {
            if (self.cached_k) |buf| {
                allocator.free(buf);
                self.cached_k = null;
            }
        }

        if (self.cached_v) |buf| allocator.free(buf);
        const v = try matmul(allocator, x, self.w_v, seq_len, self.d_model, self.d_model);
        self.cached_v = v;
        errdefer {
            if (self.cached_v) |buf| {
                allocator.free(buf);
                self.cached_v = null;
            }
        }

        if (self.cached_attn_weights) |buf| allocator.free(buf);
        const weights_len = self.n_heads * seq_len * seq_len;
        const attn_weights = try allocator.alloc(f32, weights_len);
        self.cached_attn_weights = attn_weights;
        errdefer {
            if (self.cached_attn_weights) |buf| {
                allocator.free(buf);
                self.cached_attn_weights = null;
            }
        }

        if (self.cached_attn_output) |buf| allocator.free(buf);
        const attn_concat = try allocator.alloc(f32, seq_len * self.d_model);
        self.cached_attn_output = attn_concat;
        errdefer {
            if (self.cached_attn_output) |buf| {
                allocator.free(buf);
                self.cached_attn_output = null;
            }
        }

        var q_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(q_head);
        var k_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(k_head);
        var v_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(v_head);
        var out_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(out_head);

        for (0..self.n_heads) |h| {
            const head_offset = h * self.head_dim;

            for (0..seq_len) |i| {
                const base = i * self.d_model + head_offset;
                const slice = i * self.head_dim;
                @memcpy(q_head[slice..][0..self.head_dim], self.cached_q.?[base..][0..self.head_dim]);
                @memcpy(k_head[slice..][0..self.head_dim], self.cached_k.?[base..][0..self.head_dim]);
                @memcpy(v_head[slice..][0..self.head_dim], self.cached_v.?[base..][0..self.head_dim]);
            }

            const weight_slice = self.cached_attn_weights.?[h * seq_len * seq_len ..][0 .. seq_len * seq_len];
            try scaledDotProductAttentionWithCache(
                allocator,
                q_head,
                k_head,
                v_head,
                seq_len,
                self.head_dim,
                weight_slice,
                out_head,
            );

            for (0..seq_len) |i| {
                const base = i * self.d_model + head_offset;
                const slice = i * self.head_dim;
                @memcpy(self.cached_attn_output.?[base..][0..self.head_dim], out_head[slice..][0..self.head_dim]);
            }
        }

        const output = try matmul(
            allocator,
            self.cached_attn_output.?,
            self.w_o,
            seq_len,
            self.d_model,
            self.d_model,
        );

        return output;
    }

    pub fn backward(
        self: *MultiHeadAttention,
        allocator: std.mem.Allocator,
        grad_output: []const f32,
    ) ![]f32 {
        const input = self.cached_input orelse return error.ForwardNotCalled;
        const q = self.cached_q orelse return error.ForwardNotCalled;
        const k = self.cached_k orelse return error.ForwardNotCalled;
        const v = self.cached_v orelse return error.ForwardNotCalled;
        const attn_output = self.cached_attn_output orelse return error.ForwardNotCalled;
        const attn_weights = self.cached_attn_weights orelse return error.ForwardNotCalled;

        const seq_len = self.seq_len;
        const d_model = self.d_model;

        var grad_attn_output = try allocator.alloc(f32, seq_len * d_model);
        defer allocator.free(grad_attn_output);

        for (0..seq_len) |i| {
            for (0..d_model) |j| {
                var sum: f32 = 0;
                for (0..d_model) |k_idx| {
                    sum += grad_output[i * d_model + k_idx] * self.w_o[j * d_model + k_idx];
                }
                grad_attn_output[i * d_model + j] = sum;
            }
        }

        for (0..seq_len) |i| {
            for (0..d_model) |j| {
                for (0..d_model) |k_idx| {
                    self.grad_w_o[j * d_model + k_idx] +=
                        attn_output[i * d_model + j] * grad_output[i * d_model + k_idx];
                }
            }
        }

        var grad_q_total = try allocator.alloc(f32, seq_len * d_model);
        defer allocator.free(grad_q_total);
        @memset(grad_q_total, 0.0);

        var grad_k_total = try allocator.alloc(f32, seq_len * d_model);
        defer allocator.free(grad_k_total);
        @memset(grad_k_total, 0.0);

        var grad_v_total = try allocator.alloc(f32, seq_len * d_model);
        defer allocator.free(grad_v_total);
        @memset(grad_v_total, 0.0);

        var q_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(q_head);
        var k_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(k_head);
        var v_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(v_head);
        var grad_out_head = try allocator.alloc(f32, seq_len * self.head_dim);
        defer allocator.free(grad_out_head);

        for (0..self.n_heads) |h| {
            const head_offset = h * self.head_dim;

            for (0..seq_len) |i| {
                const base = i * d_model + head_offset;
                const slice = i * self.head_dim;
                @memcpy(q_head[slice..][0..self.head_dim], q[base..][0..self.head_dim]);
                @memcpy(k_head[slice..][0..self.head_dim], k[base..][0..self.head_dim]);
                @memcpy(v_head[slice..][0..self.head_dim], v[base..][0..self.head_dim]);
                @memcpy(grad_out_head[slice..][0..self.head_dim], grad_attn_output[base..][0..self.head_dim]);
            }

            const weight_slice = attn_weights[h * seq_len * seq_len ..][0 .. seq_len * seq_len];
            const grads = try backwardAttention(
                allocator,
                grad_out_head,
                q_head,
                k_head,
                v_head,
                weight_slice,
                seq_len,
                self.head_dim,
            );

            defer allocator.free(grads.grad_q);
            defer allocator.free(grads.grad_k);
            defer allocator.free(grads.grad_v);

            for (0..seq_len) |i| {
                const base = i * d_model + head_offset;
                const slice = i * self.head_dim;
                for (0..self.head_dim) |d_idx| {
                    grad_q_total[base + d_idx] += grads.grad_q[slice + d_idx];
                    grad_k_total[base + d_idx] += grads.grad_k[slice + d_idx];
                    grad_v_total[base + d_idx] += grads.grad_v[slice + d_idx];
                }
            }
        }

        var grad_input = try allocator.alloc(f32, seq_len * d_model);
        @memset(grad_input, 0.0);

        for (0..seq_len) |i| {
            for (0..d_model) |j| {
                var sum_q: f32 = 0;
                var sum_k: f32 = 0;
                var sum_v: f32 = 0;

                for (0..d_model) |k_idx| {
                    sum_q += grad_q_total[i * d_model + k_idx] * self.w_q[j * d_model + k_idx];
                    sum_k += grad_k_total[i * d_model + k_idx] * self.w_k[j * d_model + k_idx];
                    sum_v += grad_v_total[i * d_model + k_idx] * self.w_v[j * d_model + k_idx];
                }

                grad_input[i * d_model + j] += sum_q + sum_k + sum_v;
            }
        }

        for (0..seq_len) |i| {
            for (0..d_model) |j| {
                for (0..d_model) |k_idx| {
                    const input_val = input[i * d_model + j];
                    self.grad_w_q[j * d_model + k_idx] += input_val * grad_q_total[i * d_model + k_idx];
                    self.grad_w_k[j * d_model + k_idx] += input_val * grad_k_total[i * d_model + k_idx];
                    self.grad_w_v[j * d_model + k_idx] += input_val * grad_v_total[i * d_model + k_idx];
                }
            }
        }

        return grad_input;
    }

    pub fn zeroGrad(self: *MultiHeadAttention) void {
        @memset(self.grad_w_q, 0.0);
        @memset(self.grad_w_k, 0.0);
        @memset(self.grad_w_v, 0.0);
        @memset(self.grad_w_o, 0.0);
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
    cached_input: ?[]f32 = null,
    cached_attn_input: ?[]f32 = null,
    cached_attn_output: ?[]f32 = null,
    cached_mlp_input: ?[]f32 = null,
    seq_len: usize = 0,
    d_model: usize = 0,
    allocator: std.mem.Allocator,

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
            .allocator = allocator,
            .ln1 = ln1,
            .attn = attn,
            .ln2 = ln2,
            .mlp = mlp,
        };
    }

    pub fn deinit(self: *TransformerBlock) void {
        if (self.cached_input) |buf| self.allocator.free(buf);
        if (self.cached_attn_input) |buf| self.allocator.free(buf);
        if (self.cached_attn_output) |buf| self.allocator.free(buf);
        if (self.cached_mlp_input) |buf| self.allocator.free(buf);
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
        self.allocator = allocator;
        self.seq_len = seq_len;
        self.d_model = d_model;

        if (self.cached_input) |buf| allocator.free(buf);
        self.cached_input = try allocator.alloc(f32, x.len);
        @memcpy(self.cached_input.?, x);

        var norm1 = try allocator.alloc(f32, x.len);
        defer allocator.free(norm1);
        @memcpy(norm1, x);

        for (0..seq_len) |i| {
            const offset = i * d_model;
            try self.ln1.forward(norm1[offset..][0..d_model]);
        }

        if (self.cached_attn_input) |buf| allocator.free(buf);
        self.cached_attn_input = try allocator.alloc(f32, norm1.len);
        @memcpy(self.cached_attn_input.?, norm1);

        const attn_out = try self.attn.forward(allocator, norm1, seq_len);

        if (self.cached_attn_output) |buf| allocator.free(buf);
        self.cached_attn_output = try allocator.alloc(f32, attn_out.len);
        @memcpy(self.cached_attn_output.?, attn_out);

        var residual = try allocator.alloc(f32, x.len);
        defer allocator.free(residual);
        for (0..x.len) |i| {
            residual[i] = x[i] + attn_out[i];
        }
        allocator.free(attn_out);

        var norm2 = try allocator.alloc(f32, residual.len);
        defer allocator.free(norm2);
        @memcpy(norm2, residual);

        for (0..seq_len) |i| {
            const offset = i * d_model;
            try self.ln2.forward(norm2[offset..][0..d_model]);
        }

        if (self.cached_mlp_input) |buf| allocator.free(buf);
        self.cached_mlp_input = try allocator.alloc(f32, norm2.len);
        @memcpy(self.cached_mlp_input.?, norm2);

        const mlp_out = try self.mlp.forward(allocator, norm2, seq_len);
        defer allocator.free(mlp_out);

        const output = try allocator.alloc(f32, residual.len);
        for (0..residual.len) |i| {
            output[i] = residual[i] + mlp_out[i];
        }

        return output;
    }

    pub fn backward(
        self: *TransformerBlock,
        allocator: std.mem.Allocator,
        grad_output: []const f32,
    ) ![]f32 {
        if (self.seq_len == 0 or self.d_model == 0) return error.ForwardNotCalled;

        const seq_len = self.seq_len;
        const d_model = self.d_model;
        const total = seq_len * d_model;

        var grad_residual = try allocator.alloc(f32, total);
        defer allocator.free(grad_residual);
        @memcpy(grad_residual, grad_output);

        const grad_mlp_input = try self.mlp.backward(allocator, grad_output);
        defer allocator.free(grad_mlp_input);

        var grad_ln2_input = try allocator.alloc(f32, total);
        defer allocator.free(grad_ln2_input);

        for (0..seq_len) |i| {
            const offset = i * d_model;
            const grad_token = grad_mlp_input[offset..][0..d_model];
            const grad_before = try self.ln2.backward(allocator, grad_token);
            @memcpy(grad_ln2_input[offset..][0..d_model], grad_before);
            allocator.free(grad_before);
        }

        var grad_input = try allocator.alloc(f32, total);
        errdefer allocator.free(grad_input);
        for (0..total) |i| {
            grad_residual[i] += grad_ln2_input[i];
            grad_input[i] = grad_residual[i];
        }

        const grad_attn_input_norm = try self.attn.backward(allocator, grad_residual);
        defer allocator.free(grad_attn_input_norm);

        for (0..seq_len) |i| {
            const offset = i * d_model;
            const grad_token = grad_attn_input_norm[offset..][0..d_model];
            const grad_before = try self.ln1.backward(allocator, grad_token);
            for (0..d_model) |j| {
                grad_input[offset + j] += grad_before[j];
            }
            allocator.free(grad_before);
        }

        return grad_input;
    }

    pub fn zeroGrad(self: *TransformerBlock) void {
        self.ln1.zeroGrad();
        self.attn.zeroGrad();
        self.ln2.zeroGrad();
        self.mlp.zeroGrad();
    }
};

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

    const maybe_pool = global_thread_pool;
    const work_size = M * N;
    const parallel_threshold: usize = 8192;

    if (maybe_pool) |pool| parallel_block: {
        const worker_count = pool.threadCount();
        if (worker_count > 1 and work_size >= parallel_threshold and M >= 2) {
            const MatmulContext = struct {
                a: []const f32,
                b: []const f32,
                c: []f32,
                K: usize,
                N: usize,
            };

            var ctx = MatmulContext{
                .a = a,
                .b = b,
                .c = c,
                .K = K,
                .N = N,
            };

            const rows_per_job = @max(@as(usize, 1), (M + worker_count - 1) / worker_count);
            const job_capacity = @min(worker_count, (M + rows_per_job - 1) / rows_per_job);

            const Job = struct {
                ctx: *MatmulContext,
                start_row: usize,
                end_row: usize,
            };

            const jobs = allocator.alloc(Job, job_capacity) catch break :parallel_block;
            defer allocator.free(jobs);

            const Runner = struct {
                fn run(job_ptr: *Job) void {
                    const job = job_ptr.*;
                    const context = job.ctx;
                    var row = job.start_row;
                    while (row < job.end_row) : (row += 1) {
                        const row_offset = row * context.N;
                        for (0..context.N) |j| {
                            var sum: f32 = 0;
                            const a_row = context.a[row * context.K ..];
                            const b_col_index = j;
                            for (0..context.K) |k| {
                                sum += a_row[k] * context.b[k * context.N + b_col_index];
                            }
                            context.c[row_offset + j] = sum;
                        }
                    }
                }
            };

            var job_count: usize = 0;
            var start_row: usize = 0;
            while (start_row < M) {
                const end_row = @min(start_row + rows_per_job, M);
                jobs[job_count] = .{
                    .ctx = &ctx,
                    .start_row = start_row,
                    .end_row = end_row,
                };
                try pool.submit(Job, Runner.run, &jobs[job_count]);
                job_count += 1;
                start_row = end_row;
            }

            pool.wait();
            return c;
        }
    }

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

const AttentionGradients = struct {
    grad_q: []f32,
    grad_k: []f32,
    grad_v: []f32,
};

fn scaledDotProductAttentionWithCache(
    allocator: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    seq_len: usize,
    head_dim: usize,
    weights_out: []f32,
    output: []f32,
) !void {
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

    @memcpy(weights_out, scores);

    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            var sum: f32 = 0;
            for (0..seq_len) |j| {
                sum += scores[i * seq_len + j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }
}

fn softmaxBackward(
    grad_input: []f32,
    grad_output: []const f32,
    softmax_output: []const f32,
    seq_len: usize,
) void {
    for (0..seq_len) |i| {
        const row_offset = i * seq_len;
        const grad_row = grad_output[row_offset..][0..seq_len];
        const softmax_row = softmax_output[row_offset..][0..seq_len];
        const grad_in_row = grad_input[row_offset..][0..seq_len];

        var dot: f32 = 0;
        for (grad_row, softmax_row) |g, s| {
            dot += g * s;
        }

        for (grad_in_row, grad_row, softmax_row) |*g_in, g_out, s_out| {
            g_in.* = s_out * (g_out - dot);
        }
    }
}

fn backwardAttention(
    allocator: std.mem.Allocator,
    grad_output: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    attn_weights: []const f32,
    seq_len: usize,
    head_dim: usize,
) !AttentionGradients {
    var grad_v = try allocator.alloc(f32, seq_len * head_dim);

    for (0..seq_len) |j| {
        for (0..head_dim) |d| {
            var sum: f32 = 0;
            for (0..seq_len) |i| {
                sum += attn_weights[i * seq_len + j] * grad_output[i * head_dim + d];
            }
            grad_v[j * head_dim + d] = sum;
        }
    }

    var grad_attn_weights = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(grad_attn_weights);

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            var sum: f32 = 0;
            for (0..head_dim) |d| {
                sum += grad_output[i * head_dim + d] * v[j * head_dim + d];
            }
            grad_attn_weights[i * seq_len + j] = sum;
        }
    }

    var grad_scores = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(grad_scores);

    softmaxBackward(grad_scores, grad_attn_weights, attn_weights, seq_len);

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            if (j > i) {
                grad_scores[i * seq_len + j] = 0;
            }
        }
    }

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (grad_scores) |*val| {
        val.* *= scale;
    }

    var grad_q = try allocator.alloc(f32, seq_len * head_dim);
    var grad_k = try allocator.alloc(f32, seq_len * head_dim);

    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            var sum_q: f32 = 0;
            for (0..seq_len) |j| {
                sum_q += grad_scores[i * seq_len + j] * k[j * head_dim + d];
            }
            grad_q[i * head_dim + d] = sum_q;
        }
    }

    for (0..seq_len) |j| {
        for (0..head_dim) |d| {
            var sum_k: f32 = 0;
            for (0..seq_len) |i| {
                sum_k += grad_scores[i * seq_len + j] * q[i * head_dim + d];
            }
            grad_k[j * head_dim + d] = sum_k;
        }
    }

    return AttentionGradients{
        .grad_q = grad_q,
        .grad_k = grad_k,
        .grad_v = grad_v,
    };
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
    grad_token_embeddings: []f32,
    grad_position_embeddings: []f32,
    grad_lm_head: []f32,
    cached_tokens: ?[]u32 = null,
    cached_final_hidden: ?[]f32 = null,
    forward_seq_len: usize = 0,
    thread_pool: ?*parallel.ThreadPool = null,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !Transformer {
        const token_embeddings = try allocator.alloc(f32, config.vocab_size * config.d_model);
        errdefer allocator.free(token_embeddings);

        const position_embeddings = try allocator.alloc(f32, config.context_length * config.d_model);
        errdefer allocator.free(position_embeddings);

        const grad_token_embeddings = try allocator.alloc(f32, config.vocab_size * config.d_model);
        errdefer allocator.free(grad_token_embeddings);

        const grad_position_embeddings = try allocator.alloc(f32, config.context_length * config.d_model);
        errdefer allocator.free(grad_position_embeddings);

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const scale = @sqrt(1.0 / @as(f32, @floatFromInt(config.d_model)));

        for (token_embeddings) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }

        for (position_embeddings) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }

        @memset(grad_token_embeddings, 0.0);
        @memset(grad_position_embeddings, 0.0);

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

        const grad_lm_head = try allocator.alloc(f32, config.d_model * config.vocab_size);
        errdefer allocator.free(grad_lm_head);

        const head_scale = @sqrt(1.0 / @as(f32, @floatFromInt(config.d_model)));
        for (lm_head) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * head_scale;
        }

        @memset(grad_lm_head, 0.0);

        return Transformer{
            .config = config,
            .allocator = allocator,
            .token_embeddings = token_embeddings,
            .position_embeddings = position_embeddings,
            .layers = layers,
            .ln_final = ln_final,
            .lm_head = lm_head,
            .grad_token_embeddings = grad_token_embeddings,
            .grad_position_embeddings = grad_position_embeddings,
            .grad_lm_head = grad_lm_head,
            .thread_pool = null,
        };
    }

    pub fn deinit(self: *Transformer) void {
        self.allocator.free(self.token_embeddings);
        self.allocator.free(self.position_embeddings);
        self.allocator.free(self.grad_token_embeddings);
        self.allocator.free(self.grad_position_embeddings);
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.ln_final.deinit();
        self.allocator.free(self.lm_head);
        self.allocator.free(self.grad_lm_head);
        if (self.cached_tokens) |buf| self.allocator.free(buf);
        if (self.cached_final_hidden) |buf| self.allocator.free(buf);
    }

    pub fn setThreadPool(self: *Transformer, pool: ?*parallel.ThreadPool) void {
        self.thread_pool = pool;
    }

    pub fn forward(self: *Transformer, tokens: []const u32) ![]f32 {
        const seq_len = tokens.len;
        const d_model = self.config.d_model;
        const vocab_size = self.config.vocab_size;

        const prev_pool = global_thread_pool;
        if (self.thread_pool) |pool| {
            global_thread_pool = pool;
        }
        defer global_thread_pool = prev_pool;

        self.forward_seq_len = seq_len;

        if (self.cached_tokens) |old| self.allocator.free(old);
        self.cached_tokens = try self.allocator.alloc(u32, tokens.len);
        @memcpy(self.cached_tokens.?, tokens);

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

        if (self.cached_final_hidden) |old| self.allocator.free(old);
        self.cached_final_hidden = try self.allocator.alloc(f32, hidden.len);
        @memcpy(self.cached_final_hidden.?, hidden);

        const logits = try matmul(self.allocator, hidden, self.lm_head, seq_len, d_model, vocab_size);
        self.allocator.free(hidden);
        return logits;
    }

    pub fn backward(self: *Transformer, grad_logits: []const f32) !void {
        const tokens = self.cached_tokens orelse return error.ForwardNotCalled;
        const final_hidden = self.cached_final_hidden orelse return error.ForwardNotCalled;
        const seq_len = self.forward_seq_len;
        if (seq_len == 0) return error.ForwardNotCalled;

        const prev_pool = global_thread_pool;
        if (self.thread_pool) |pool| {
            global_thread_pool = pool;
        }
        defer global_thread_pool = prev_pool;

        const d_model = self.config.d_model;
        const vocab_size = self.config.vocab_size;

        for (0..seq_len) |i| {
            for (0..d_model) |j| {
                for (0..vocab_size) |k| {
                    self.grad_lm_head[j * vocab_size + k] +=
                        final_hidden[i * d_model + j] * grad_logits[i * vocab_size + k];
                }
            }
        }

        var grad_hidden = try self.allocator.alloc(f32, seq_len * d_model);
        defer self.allocator.free(grad_hidden);

        for (0..seq_len) |i| {
            for (0..d_model) |j| {
                var sum: f32 = 0;
                for (0..vocab_size) |k| {
                    sum += grad_logits[i * vocab_size + k] * self.lm_head[j * vocab_size + k];
                }
                grad_hidden[i * d_model + j] = sum;
            }
        }

        var grad_before_ln = try self.allocator.alloc(f32, seq_len * d_model);

        for (0..seq_len) |i| {
            const offset = i * d_model;
            const grad_slice = grad_hidden[offset..][0..d_model];
            const grad_in = try self.ln_final.backward(self.allocator, grad_slice);
            @memcpy(grad_before_ln[offset..][0..d_model], grad_in);
            self.allocator.free(grad_in);
        }

        var current_grad = grad_before_ln;
        var idx = self.config.n_layers;
        while (idx > 0) : (idx -= 1) {
            const next_grad = try self.layers[idx - 1].backward(self.allocator, current_grad);
            self.allocator.free(current_grad);
            current_grad = next_grad;
        }

        for (tokens, 0..) |token, pos| {
            const offset = pos * d_model;
            const token_offset = token * d_model;
            const pos_offset = pos * d_model;
            for (0..d_model) |j| {
                const grad_val = current_grad[offset + j];
                self.grad_token_embeddings[token_offset + j] += grad_val;
                self.grad_position_embeddings[pos_offset + j] += grad_val;
            }
        }

        self.allocator.free(current_grad);
    }

    pub fn zeroGrad(self: *Transformer) void {
        @memset(self.grad_token_embeddings, 0.0);
        @memset(self.grad_position_embeddings, 0.0);
        @memset(self.grad_lm_head, 0.0);
        for (self.layers) |*layer| layer.zeroGrad();
        self.ln_final.zeroGrad();
    }
};
