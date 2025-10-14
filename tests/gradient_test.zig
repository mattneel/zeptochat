const std = @import("std");
const testing = std.testing;
const transformer = @import("transformer");
const LayerNorm = transformer.LayerNorm;
const MLP = transformer.MLP;
const MultiHeadAttention = transformer.MultiHeadAttention;
const Transformer = transformer.Transformer;
const TinyConfig = transformer.TinyConfig;

fn numericalGradient(
    param: *f32,
    eps: f32,
    comptime LossFn: type,
    func: fn (LossFn) anyerror!f32,
    loss_fn: LossFn,
) !f32 {
    const original = param.*;

    param.* = original + eps;
    const loss_plus = try func(loss_fn);

    param.* = original - eps;
    const loss_minus = try func(loss_fn);

    param.* = original;

    return (loss_plus - loss_minus) / (2.0 * eps);
}

test "layernorm gradient check weight" {
    var ln = try LayerNorm.init(testing.allocator, 4);
    defer ln.deinit();

    var input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try ln.forward(&input);

    const LossFn = struct {
        ln_ptr: *LayerNorm,
        data: [4]f32,

        pub fn call(self: @This()) !f32 {
            var copy = self.data;
            try self.ln_ptr.forward(&copy);
            var sum: f32 = 0;
            for (copy) |val| sum += val;
            return sum;
        }
    };

    const loss_fn = LossFn{ .ln_ptr = &ln, .data = input };

    var grad_output = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const grad_input = try ln.backward(testing.allocator, &grad_output);
    defer testing.allocator.free(grad_input);

    const numerical = try numericalGradient(&ln.weight[0], 1e-4, LossFn, LossFn.call, loss_fn);
    const analytical = ln.grad_weight[0];

    const diff = @abs(numerical - analytical);
    const relative_error = diff / (@abs(numerical) + @abs(analytical) + 1e-8);

    try testing.expect(relative_error < 1e-3);
}

test "mlp gradient check weight" {
    const d_model = 8;
    const seq_len = 2;

    var mlp = try MLP.init(testing.allocator, d_model);
    defer mlp.deinit();

    const input = try testing.allocator.alloc(f32, seq_len * d_model);
    defer testing.allocator.free(input);

    for (input, 0..) |*val, idx| {
        val.* = @as(f32, @floatFromInt(idx)) / 10.0;
    }

    const output = try mlp.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(output);

    mlp.zeroGrad();

    const grad_output = try testing.allocator.alloc(f32, output.len);
    defer testing.allocator.free(grad_output);
    @memset(grad_output, 1.0);

    const grad_input = try mlp.backward(testing.allocator, grad_output);
    defer testing.allocator.free(grad_input);

    const eps: f32 = 1e-4;
    const param = &mlp.w1[0];
    const original = param.*;

    param.* = original + eps;
    const out_plus = try mlp.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(out_plus);
    var loss_plus: f32 = 0;
    for (out_plus) |val| loss_plus += val;

    param.* = original - eps;
    const out_minus = try mlp.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(out_minus);
    var loss_minus: f32 = 0;
    for (out_minus) |val| loss_minus += val;

    param.* = original;

    const numerical = (loss_plus - loss_minus) / (2.0 * eps);
    const analytical = mlp.grad_w1[0];

    const diff = @abs(numerical - analytical);
    const relative_error = diff / (@abs(numerical) + @abs(analytical) + 1e-8);

    try testing.expect(relative_error < 1e-2);
}

test "attention gradient check weight" {
    const d_model = 8;
    const n_heads = 2;
    const seq_len = 3;

    var attn = try MultiHeadAttention.init(testing.allocator, d_model, n_heads);
    defer attn.deinit();

    const input = try testing.allocator.alloc(f32, seq_len * d_model);
    defer testing.allocator.free(input);

    for (input, 0..) |*val, idx| {
        val.* = @as(f32, @floatFromInt(idx)) / 20.0;
    }

    const output = try attn.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(output);

    attn.zeroGrad();

    const grad_output = try testing.allocator.alloc(f32, output.len);
    defer testing.allocator.free(grad_output);
    @memset(grad_output, 1.0);

    const grad_input = try attn.backward(testing.allocator, grad_output);
    defer testing.allocator.free(grad_input);

    const eps: f32 = 1e-4;
    const param = &attn.w_q[0];
    const original = param.*;

    param.* = original + eps;
    const out_plus = try attn.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(out_plus);
    var loss_plus: f32 = 0;
    for (out_plus) |val| loss_plus += val;

    param.* = original - eps;
    const out_minus = try attn.forward(testing.allocator, input, seq_len);
    defer testing.allocator.free(out_minus);
    var loss_minus: f32 = 0;
    for (out_minus) |val| loss_minus += val;

    param.* = original;

    const numerical = (loss_plus - loss_minus) / (2.0 * eps);
    const analytical = attn.grad_w_q[0];

    const diff = @abs(numerical - analytical);
    const relative_error = diff / (@abs(numerical) + @abs(analytical) + 1e-8);

    try testing.expect(relative_error < 2e-2);
}

test "transformer gradient check embedding weight" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const tokens = [_]u32{ 1, 2, 3 };

    model.zeroGrad();

    const logits = try model.forward(&tokens);
    defer testing.allocator.free(logits);

    const grad_logits = try testing.allocator.alloc(f32, logits.len);
    defer testing.allocator.free(grad_logits);
    @memset(grad_logits, 1.0);

    const eps: f32 = 1e-4;
    const param = &model.token_embeddings[0];
    const original = param.*;

    param.* = original + eps;
    const logits_plus = try model.forward(&tokens);
    defer testing.allocator.free(logits_plus);
    var loss_plus: f32 = 0;
    for (logits_plus) |val| loss_plus += val;

    param.* = original - eps;
    const logits_minus = try model.forward(&tokens);
    defer testing.allocator.free(logits_minus);
    var loss_minus: f32 = 0;
    for (logits_minus) |val| loss_minus += val;

    const numerical = (loss_plus - loss_minus) / (2.0 * eps);

    param.* = original;

    model.zeroGrad();
    const logits_final = try model.forward(&tokens);
    defer testing.allocator.free(logits_final);
    try model.backward(grad_logits);

    const analytical = model.grad_token_embeddings[0];

    const diff = @abs(numerical - analytical);
    const relative_error = diff / (@abs(numerical) + @abs(analytical) + 1e-8);

    try testing.expect(relative_error < 5e-2);
}
