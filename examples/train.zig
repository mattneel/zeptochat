const std = @import("std");
const transformer = @import("transformer");
const dataset_mod = @import("dataset");
const optimizer_mod = @import("optimizer");
const training = @import("training");

const Transformer = transformer.Transformer;
const TinyConfig = transformer.TinyConfig;
const Dataset = dataset_mod.Dataset;
const AdamW = optimizer_mod.AdamW;
const crossEntropyLoss = training.crossEntropyLoss;
const crossEntropyGrad = training.crossEntropyGrad;

const TrainError = error{
    InvalidInteger,
    InvalidFloat,
    InvalidSequenceLength,
};

const default_tokens_path = "data/train.tokens";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.skip(); // executable name
    const tokens_path = args.next() orelse default_tokens_path;

    const epochs = try parseOptionalInt(args.next(), 3);
    const seq_len = try parseOptionalInt(args.next(), TinyConfig.context_length);
    const lr = try parseOptionalFloat(args.next(), 0.001);

    if (seq_len == 0 or seq_len > TinyConfig.context_length) {
        std.log.err(
            "seq_len must be in 1..={d} (got {d})",
            .{ TinyConfig.context_length, seq_len },
        );
        return TrainError.InvalidSequenceLength;
    }

    std.log.info("Loading tokens from {s}…", .{tokens_path});
    var dataset = try Dataset.init(allocator, tokens_path);
    defer dataset.deinit();
    std.log.info("Loaded {d} tokens", .{dataset.tokens.len});

    var model = try Transformer.init(allocator, TinyConfig);
    defer model.deinit();

    const param_count = AdamW.countParams(&model);
    std.log.info("Model parameters: {d}", .{param_count});
    var optimizer = try AdamW.init(allocator, param_count);
    defer optimizer.deinit();

    model.zeroGrad();

    const batch_size: usize = 1; // current transformer expects flat sequence
    var global_step: usize = 0;

    for (0..epochs) |epoch| {
        std.log.info("=== Epoch {}/{} ===", .{ epoch + 1, epochs });

        var iter = dataset.iterator(batch_size, seq_len);
        var loss_sum: f32 = 0;
        var loss_count: usize = 0;

        while (iter.next()) |batch| {
            global_step += 1;

            const logits = try model.forward(batch.tokens);
            defer allocator.free(logits);

            const loss = crossEntropyLoss(
                logits,
                batch.targets,
                batch.tokens.len,
                TinyConfig.vocab_size,
            );
            loss_sum += loss;
            loss_count += 1;

            const grad_logits = try crossEntropyGrad(
                allocator,
                logits,
                batch.targets,
                batch.tokens.len,
                TinyConfig.vocab_size,
            );
            defer allocator.free(grad_logits);

            try model.backward(grad_logits);
            optimizer.step(&model, lr);
            model.zeroGrad();

            if (global_step % 50 == 0) {
                const avg_loss = loss_sum / @as(f32, @floatFromInt(loss_count));
                std.log.info("step {d}: avg loss {d:.4}", .{ global_step, avg_loss });
                loss_sum = 0;
                loss_count = 0;
            }
        }

        iter.reset();

        if (loss_count != 0) {
            const avg = loss_sum / @as(f32, @floatFromInt(loss_count));
            std.log.info("epoch {}/{} average loss {d:.4}", .{ epoch + 1, epochs, avg });
        }
    }

    std.log.info("Training complete", .{});
}

fn parseOptionalInt(maybe: ?[]const u8, default_value: usize) TrainError!usize {
    if (maybe) |value| {
        return std.fmt.parseInt(usize, value, 10) catch return TrainError.InvalidInteger;
    }
    return default_value;
}

fn parseOptionalFloat(maybe: ?[]const u8, default_value: f32) TrainError!f32 {
    if (maybe) |value| {
        return std.fmt.parseFloat(f32, value) catch return TrainError.InvalidFloat;
    }
    return default_value;
}
