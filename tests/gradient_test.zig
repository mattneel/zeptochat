const std = @import("std");
const testing = std.testing;
const LayerNorm = @import("transformer").LayerNorm;

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

    var mlp = try @import("transformer").MLP.init(testing.allocator, d_model);
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
