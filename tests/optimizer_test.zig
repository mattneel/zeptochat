const std = @import("std");
const testing = std.testing;
const transformer = @import("transformer");
const optimizer_mod = @import("optimizer");

const Transformer = transformer.Transformer;
const TinyConfig = transformer.TinyConfig;
const AdamW = optimizer_mod.AdamW;

test "adamw updates parameters and moments" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const num_params = AdamW.countParams(&model);
    var opt = try AdamW.init(testing.allocator, num_params);
    defer opt.deinit();

    // Seed a single gradient
    const grad_value: f32 = 1.0;
    model.zeroGrad();
    model.grad_token_embeddings[0] = grad_value;

    const before = model.token_embeddings[0];
    const lr: f32 = 0.001;

    opt.step(&model, lr);

    try testing.expectEqual(@as(usize, 1), opt.step_count);

    const beta1 = opt.beta1;
    const beta2 = opt.beta2;
    const bias_correction1 = 1.0 - std.math.pow(f32, beta1, 1.0);
    const bias_correction2 = 1.0 - std.math.pow(f32, beta2, 1.0);
    const adjusted_lr = lr * @sqrt(bias_correction2) / bias_correction1;

    const expected_m = (1.0 - beta1) * grad_value;
    const expected_v = (1.0 - beta2) * grad_value * grad_value;

    try testing.expectApproxEqAbs(expected_m, opt.m[0], 1e-6);
    try testing.expectApproxEqAbs(expected_v, opt.v[0], 1e-6);

    const decay_factor = 1.0 - adjusted_lr * opt.weight_decay;
    const expected_after = before * decay_factor -
        adjusted_lr * expected_m / (@sqrt(expected_v) + opt.eps);

    try testing.expectApproxEqAbs(expected_after, model.token_embeddings[0], 1e-5);
}
