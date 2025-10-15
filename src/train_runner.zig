const std = @import("std");
const transformer = @import("transformer");
const dataset_mod = @import("dataset");
const optimizer_mod = @import("optimizer");
const training = @import("training");
const checkpoint = @import("checkpoint");

const Transformer = transformer.Transformer;
const TinyConfig = transformer.TinyConfig;
const Dataset = dataset_mod.Dataset;
const AdamW = optimizer_mod.AdamW;

pub const TrainConfig = struct {
    tokens_path: []const u8,
    epochs: usize = 3,
    seq_len: usize = 64,
    learning_rate: f32 = 0.001,
    batch_size: usize = 1,
    checkpoint_dir: []const u8 = "checkpoints",
    checkpoint_prefix: []const u8 = "model",
    checkpoint_frequency: usize = 1, // epochs
};

pub fn run(allocator: std.mem.Allocator, config: TrainConfig) !void {
    if (config.seq_len == 0 or config.seq_len > TinyConfig.context_length) {
        std.log.err(
            "seq_len must be between 1 and {d}, got {d}",
            .{ TinyConfig.context_length, config.seq_len },
        );
        return error.InvalidSequenceLength;
    }

    std.log.info("Loading tokens from {s}…", .{config.tokens_path});
    var dataset = try Dataset.init(allocator, config.tokens_path);
    defer dataset.deinit();
    std.log.info("Loaded {d} tokens", .{dataset.tokens.len});

    var model = try Transformer.init(allocator, TinyConfig);
    defer model.deinit();

    const total_params = AdamW.countParams(&model);
    std.log.info("Model parameters: {d}", .{total_params});
    var optimizer = try AdamW.init(allocator, total_params);
    defer optimizer.deinit();

    model.zeroGrad();

    try std.fs.cwd().makePath(config.checkpoint_dir);

    var global_step: usize = 0;

    for (0..config.epochs) |epoch_idx| {
        std.log.info("=== Epoch {}/{} ===", .{ epoch_idx + 1, config.epochs });
        var iter = dataset.iterator(config.batch_size, config.seq_len);

        var loss_sum: f32 = 0;
        var loss_count: usize = 0;

        while (iter.next()) |batch| {
            global_step += 1;

            const logits = try model.forward(batch.tokens);
            defer allocator.free(logits);

            const loss = training.crossEntropyLoss(
                logits,
                batch.targets,
                batch.tokens.len,
                TinyConfig.vocab_size,
            );

            loss_sum += loss;
            loss_count += 1;

            const grad_logits = try training.crossEntropyGrad(
                allocator,
                logits,
                batch.targets,
                batch.tokens.len,
                TinyConfig.vocab_size,
            );
            defer allocator.free(grad_logits);

            try model.backward(grad_logits);
            optimizer.step(&model, config.learning_rate);
            model.zeroGrad();

            if (global_step % 50 == 0) {
                const avg_loss = loss_sum / @as(f32, @floatFromInt(loss_count));
                std.log.info("step {d}: avg loss {d:.4}", .{ global_step, avg_loss });
                loss_sum = 0;
                loss_count = 0;
            }
        }

        if (loss_count != 0) {
            const avg_loss = loss_sum / @as(f32, @floatFromInt(loss_count));
            std.log.info(
                "epoch {}/{} average loss {d:.4}",
                .{ epoch_idx + 1, config.epochs, avg_loss },
            );
        }

        if ((epoch_idx + 1) % config.checkpoint_frequency == 0) {
            const file_name = try formatCheckpointName(
                allocator,
                config.checkpoint_dir,
                config.checkpoint_prefix,
                epoch_idx + 1,
            );
            defer allocator.free(file_name);

            std.log.info("Saving checkpoint to {s}", .{file_name});
            try checkpoint.save(&model, file_name, global_step);
        }
    }
}

fn formatCheckpointName(
    allocator: std.mem.Allocator,
    dir: []const u8,
    prefix: []const u8,
    epoch: usize,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}_epoch{d}.ckpt", .{ dir, prefix, epoch });
}
