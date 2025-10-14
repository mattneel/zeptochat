const std = @import("std");
const testing = std.testing;
const Transformer = @import("transformer").Transformer;
const TinyConfig = @import("transformer").TinyConfig;
const LayerNorm = @import("transformer").LayerNorm;
const MultiHeadAttention = @import("transformer").MultiHeadAttention;

test "transformer init allocates embeddings" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    try testing.expectEqual(TinyConfig.vocab_size * TinyConfig.d_model, model.token_embeddings.len);
    try testing.expectEqual(TinyConfig.context_length * TinyConfig.d_model, model.position_embeddings.len);
}

test "transformer forward returns correct shape" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const tokens = [_]u32{ 1, 2, 3, 4 };
    const hidden = try model.forward(&tokens);
    defer testing.allocator.free(hidden);

    try testing.expectEqual(tokens.len * TinyConfig.d_model, hidden.len);
}

test "transformer embeddings are randomized" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const d_model = TinyConfig.d_model;
    const token0 = model.token_embeddings[0..d_model];
    const token1 = model.token_embeddings[d_model .. d_model * 2];

    var identical = true;
    for (token0, token1) |a, b| {
        if (a != b) {
            identical = false;
            break;
        }
    }

    try testing.expect(!identical);
}

test "layernorm init sets affine parameters" {
    var ln = try LayerNorm.init(testing.allocator, 64);
    defer ln.deinit();

    for (ln.weight) |w| try testing.expectEqual(@as(f32, 1.0), w);
    for (ln.bias) |b| try testing.expectEqual(@as(f32, 0.0), b);
}

test "layernorm normalises to zero mean unit variance" {
    var ln = try LayerNorm.init(testing.allocator, 4);
    defer ln.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    ln.forward(&data);

    var sum: f32 = 0;
    for (data) |val| sum += val;
    const mean = sum / 4.0;
    try testing.expect(@abs(mean) < 1e-5);

    var var_sum: f32 = 0;
    for (data) |val| {
        const diff = val - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / 4.0;
    try testing.expect(@abs(variance - 1.0) < 1e-4);
}

test "layernorm applies affine transform" {
    var ln = try LayerNorm.init(testing.allocator, 4);
    defer ln.deinit();

    ln.weight[0] = 2.0;
    ln.bias[0] = 3.0;

    var data = [_]f32{ 1.5, 1.5, 1.5, 1.5 };
    ln.forward(&data);

    try testing.expect(!std.math.isNan(data[0]));
    try testing.expect(!std.math.isInf(data[0]));
}

test "layernorm handles zero variance" {
    var ln = try LayerNorm.init(testing.allocator, 4);
    defer ln.deinit();

    var data = [_]f32{ 5.0, 5.0, 5.0, 5.0 };
    ln.forward(&data);

    for (data) |val| {
        try testing.expect(!std.math.isNan(val));
        try testing.expect(!std.math.isInf(val));
    }
}

test "attention init allocates projections" {
    const d_model = 64;
    const n_heads = 4;
    const d_squared = d_model * d_model;

    var attn = try MultiHeadAttention.init(testing.allocator, d_model, n_heads);
    defer attn.deinit();

    try testing.expectEqual(d_model, attn.d_model);
    try testing.expectEqual(n_heads, attn.n_heads);
    try testing.expectEqual(d_model / n_heads, attn.head_dim);

    try testing.expectEqual(d_squared, attn.w_q.len);
    try testing.expectEqual(d_squared, attn.w_k.len);
    try testing.expectEqual(d_squared, attn.w_v.len);
    try testing.expectEqual(d_squared, attn.w_o.len);
}

test "attention forward preserves shape" {
    const d_model = 64;
    const n_heads = 4;
    const seq_len = 8;

    var attn = try MultiHeadAttention.init(testing.allocator, d_model, n_heads);
    defer attn.deinit();

    const input = try testing.allocator.alloc(f32, seq_len * d_model);
    defer testing.allocator.free(input);
    @memset(input, 1.0);

    const output = try attn.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(output);

    try testing.expectEqual(input.len, output.len);
}

test "attention applies causal mask" {
    const d_model = 16;
    const n_heads = 4;
    const seq_len = 4;

    var attn = try MultiHeadAttention.init(testing.allocator, d_model, n_heads);
    defer attn.deinit();

    const input = try testing.allocator.alloc(f32, seq_len * d_model);
    defer testing.allocator.free(input);

    for (input, 0..) |*val, idx| {
        val.* = @floatFromInt(idx);
    }

    const output = try attn.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(output);

    var different = false;
    for (input, output) |in_val, out_val| {
        if (@abs(in_val - out_val) > 1e-6) {
            different = true;
            break;
        }
    }

    try testing.expect(different);
}
