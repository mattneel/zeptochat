const std = @import("std");
const testing = std.testing;
const transformer = @import("transformer");
const training = @import("training");

const Transformer = transformer.Transformer;
const TinyConfig = transformer.TinyConfig;

const crossEntropyLoss = training.crossEntropyLoss;
const crossEntropyGrad = training.crossEntropyGrad;
const sgdStep = training.sgdStep;

test "training: overfit single batch" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const targets = [_]u32{ 2, 3, 4, 5, 6 };

    const learning_rate: f32 = 0.01;
    const num_steps: usize = 200;

    var initial_loss: f32 = 0;
    var final_loss: f32 = 0;

    model.zeroGrad();

    for (0..num_steps) |step| {
        const logits = try model.forward(&tokens);
        defer testing.allocator.free(logits);

        const loss = crossEntropyLoss(logits, &targets, tokens.len, TinyConfig.vocab_size);

        if (step == 0) {
            initial_loss = loss;
            std.debug.print("\nOverfit test - Initial loss: {d:.4}\n", .{loss});
        }
        if (step % 50 == 0) {
            std.debug.print("Step {d:3}, Loss: {d:.4}\n", .{ step, loss });
        }

        const grad_logits = try crossEntropyGrad(
            testing.allocator,
            logits,
            &targets,
            tokens.len,
            TinyConfig.vocab_size,
        );
        defer testing.allocator.free(grad_logits);

        try model.backward(grad_logits);
        sgdStep(&model, learning_rate);
        model.zeroGrad();

        if (step == num_steps - 1) {
            final_loss = loss;
        }
    }

    std.debug.print("Final loss: {d:.4}\n", .{final_loss});
    std.debug.print("Loss reduction: {d:.2}x\n", .{initial_loss / final_loss});

    try testing.expect(final_loss < initial_loss / 5.0);
    try testing.expect(final_loss < 1.0);
}

test "training: loss trend decreases" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const tokens = [_]u32{ 1, 2, 3, 4 };
    const targets = [_]u32{ 2, 3, 4, 5 };

    const learning_rate: f32 = 0.01;
    const steps: usize = 60;

    var losses = try testing.allocator.alloc(f32, steps);
    defer testing.allocator.free(losses);

    model.zeroGrad();

    for (0..steps) |step| {
        const logits = try model.forward(&tokens);
        defer testing.allocator.free(logits);

        const loss = crossEntropyLoss(logits, &targets, tokens.len, TinyConfig.vocab_size);
        losses[step] = loss;

        const grad_logits = try crossEntropyGrad(
            testing.allocator,
            logits,
            &targets,
            tokens.len,
            TinyConfig.vocab_size,
        );
        defer testing.allocator.free(grad_logits);

        try model.backward(grad_logits);
        sgdStep(&model, learning_rate);
        model.zeroGrad();
    }

    const window = 10;
    const inv_window = 1.0 / @as(f32, @floatFromInt(window));

    var first_avg: f32 = 0;
    for (0..window) |i| {
        first_avg += losses[i];
    }
    first_avg *= inv_window;

    var last_avg: f32 = 0;
    for (0..window) |i| {
        last_avg += losses[steps - window + i];
    }
    last_avg *= inv_window;

    std.debug.print("\nFirst avg loss: {d:.4}\n", .{first_avg});
    std.debug.print("Last avg loss: {d:.4}\n", .{last_avg});

    try testing.expect(last_avg < first_avg);
}
