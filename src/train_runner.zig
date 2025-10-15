const std = @import("std");
const transformer = @import("transformer");
const dataset_mod = @import("dataset");
const optimizer_mod = @import("optimizer");
const training = @import("training");
const checkpoint = @import("checkpoint");
const parallel = @import("parallel");

const Transformer = transformer.Transformer;
const TinyConfig = transformer.TinyConfig;
const Dataset = dataset_mod.Dataset;
const AdamW = optimizer_mod.AdamW;

const ns_per_s = std.time.ns_per_s;
const ns_per_s_f = @as(f64, @floatFromInt(ns_per_s));

pub const TrainConfig = struct {
    tokens_path: []const u8,
    epochs: usize = 3,
    seq_len: usize = TinyConfig.context_length,
    learning_rate: f32 = 0.001,
    batch_size: usize = 1,
    checkpoint_dir: []const u8 = "checkpoints",
    checkpoint_prefix: []const u8 = "model",
    checkpoint_frequency: usize = 1, // epochs
    model: transformer.ModelConfig = transformer.TinyConfig,
};

pub fn run(allocator: std.mem.Allocator, config: TrainConfig) !void {
    if (config.batch_size == 0) {
        std.log.err("batch_size must be greater than zero", .{});
        return error.InvalidBatchConfiguration;
    }

    const model_config = config.model;
    var seq_len = config.seq_len;
    if (seq_len == 0) {
        seq_len = model_config.context_length;
    }
    if (seq_len > model_config.context_length) {
        std.log.warn(
            "seq_len {d} exceeds model context length {d}; clamping to {d}",
            .{ seq_len, model_config.context_length, model_config.context_length },
        );
        seq_len = model_config.context_length;
    }

    std.log.info("Loading tokens from {s}…", .{config.tokens_path});
    var dataset = try Dataset.init(allocator, config.tokens_path);
    defer dataset.deinit();
    std.log.info("Loaded {d} tokens", .{dataset.tokens.len});

    std.log.info(
        "Model config: vocab={d}, context={d}, d_model={d}, heads={d}, layers={d}, dropout={d:.3}",
        .{
            model_config.vocab_size,
            model_config.context_length,
            model_config.d_model,
            model_config.n_heads,
            model_config.n_layers,
            model_config.dropout,
        },
    );

    var model = try Transformer.init(allocator, model_config);
    defer model.deinit();

    const total_params = AdamW.countParams(&model);
    std.log.info("Model parameters: {d}", .{total_params});
    var optimizer = try AdamW.init(allocator, total_params);
    defer optimizer.deinit();

    model.zeroGrad();

    var thread_pool = try parallel.ThreadPool.init(allocator, 0);
    defer thread_pool.deinit();
    model.setThreadPool(&thread_pool);

    try std.fs.cwd().makePath(config.checkpoint_dir);

    const tokens_per_batch = config.batch_size * seq_len;
    if (tokens_per_batch == 0) {
        std.log.err(
            "batch_size ({d}) × seq_len ({d}) is zero; cannot create batches",
            .{ config.batch_size, seq_len },
        );
        return error.InvalidBatchConfiguration;
    }

    if (dataset.tokens.len <= 1 or dataset.tokens.len <= tokens_per_batch) {
        std.log.err(
            "dataset has {d} tokens; need at least {d} to form one batch",
            .{ dataset.tokens.len, tokens_per_batch + 1 },
        );
        return error.DatasetTooSmall;
    }

    const available_tokens = dataset.tokens.len - 1;
    const steps_per_epoch = available_tokens / tokens_per_batch;
    if (steps_per_epoch == 0) {
        std.log.err(
            "sequence/batch configuration yields zero steps per epoch (tokens_per_batch={d})",
            .{tokens_per_batch},
        );
        return error.NoSteps;
    }

    const total_steps = steps_per_epoch * config.epochs;
    if (total_steps == 0) {
        std.log.err("no training steps to run (epochs={d}, steps_per_epoch={d})", .{
            config.epochs,
            steps_per_epoch,
        });
        return error.NoSteps;
    }

    const steps_log_interval = @max(@as(usize, 100), steps_per_epoch / 20);
    const label_update_period_ns: u64 = ns_per_s;

    std.log.info(
        "Batch size {d}, seq_len {d}, tokens/batch {d}",
        .{ config.batch_size, seq_len, tokens_per_batch },
    );
    std.log.info(
        "Steps per epoch: {d} (total steps {d})",
        .{ steps_per_epoch, total_steps },
    );
    std.log.info(
        "Checkpointing every {d} epoch(s) to {s}",
        .{ config.checkpoint_frequency, config.checkpoint_dir },
    );

    var global_step: usize = 0;
    var timer = try std.time.Timer.start();
    const progress_root = std.Progress.start(.{
        .root_name = "train",
        .estimated_total_items = total_steps,
    });
    defer progress_root.end();
    progress_root.setName("loss ---- tok/s ---- t ----s");

    var progress_label_buf: [std.Progress.Node.max_name_len]u8 = undefined;
    var ema_loss: f64 = 0;
    var ema_initialized = false;
    var last_label_update_ns: u64 = 0;

    for (0..config.epochs) |epoch_idx| {
        const epoch_start_elapsed = timer.read();
        var iter = dataset.iterator(config.batch_size, seq_len);

        var epoch_node = progress_root.start("", steps_per_epoch);
        var epoch_label_buf: [std.Progress.Node.max_name_len]u8 = undefined;
        const epoch_label = std.fmt.bufPrint(
            &epoch_label_buf,
            "epoch {d}/{d}",
            .{ epoch_idx + 1, config.epochs },
        ) catch "epoch";
        epoch_node.setName(epoch_label);

        var epoch_loss_sum: f64 = 0;
        var epoch_steps: usize = 0;

        while (iter.next()) |batch| {
            global_step += 1;
            epoch_steps += 1;

            const batch_size = config.batch_size;
            const seq_len_usize = seq_len;
            const vocab_size = model_config.vocab_size;
            const batch_size_f = @as(f64, @floatFromInt(batch_size));
            const inv_batch_f32: f32 = 1.0 / @as(f32, @floatFromInt(batch_size));

            var batch_loss_sum: f64 = 0;

            var batch_node: ?std.Progress.Node = null;
            var batch_label_buf: [std.Progress.Node.max_name_len]u8 = undefined;
            if (batch_size > 1) {
                batch_node = epoch_node.start("", batch_size);
                if (batch_node) |*node| node.setEstimatedTotalItems(batch_size);
            }
            defer if (batch_node) |*node| node.end();

            for (0..batch_size) |sample_idx| {
                const offset = sample_idx * seq_len_usize;
                const sample_tokens = batch.tokens[offset..][0..seq_len_usize];
                const sample_targets = batch.targets[offset..][0..seq_len_usize];

                if (batch_node) |*node| {
                    node.setName(std.fmt.bufPrint(
                        &batch_label_buf,
                        "sample {d}/{d}",
                        .{ sample_idx + 1, batch_size },
                    ) catch "sample");
                }

                const preview_now = timer.read();
                const preview_elapsed = @as(f64, @floatFromInt(preview_now)) / ns_per_s_f;
                if (preview_elapsed >= 0) {
                    const progress_label = std.fmt.bufPrint(
                        &progress_label_buf,
                        "loss ---- tok/s ---- step {d}/{d} ({d}/{d})",
                        .{ global_step, total_steps, sample_idx + 1, batch_size },
                    ) catch "training";
                    progress_root.setName(progress_label);
                    last_label_update_ns = preview_now;
                }

                const logits = try model.forward(sample_tokens);
                defer allocator.free(logits);

                const loss_f32 = training.crossEntropyLoss(
                    logits,
                    sample_targets,
                    seq_len_usize,
                    vocab_size,
                );
                batch_loss_sum += @as(f64, loss_f32);

                const grad_logits = try training.crossEntropyGrad(
                    allocator,
                    logits,
                    sample_targets,
                    seq_len_usize,
                    vocab_size,
                );
                defer allocator.free(grad_logits);

                if (batch_size > 1) {
                    for (grad_logits) |*g| g.* *= inv_batch_f32;
                }

                try model.backward(grad_logits);

                if (batch_node) |*node| {
                    node.setCompletedItems(sample_idx + 1);
                    node.setName(std.fmt.bufPrint(
                        &batch_label_buf,
                        "sample {d}/{d}",
                        .{ sample_idx + 1, batch_size },
                    ) catch "sample");
                }

                std.Progress.maybeRefresh();

                const now_ns_sample = timer.read();
                const since_last_sample = if (last_label_update_ns == 0) now_ns_sample else now_ns_sample - last_label_update_ns;
                if (last_label_update_ns == 0 or since_last_sample >= label_update_period_ns) {
                    const elapsed_ns_f = @as(f64, @floatFromInt(now_ns_sample));
                    if (elapsed_ns_f > 0) {
                        const elapsed_s = elapsed_ns_f / ns_per_s_f;
                        const steps_f = @as(f64, @floatFromInt(global_step - 1)) +
                            (@as(f64, @floatFromInt(sample_idx + 1)) / @as(f64, @floatFromInt(batch_size)));
                        const batch_tokens_f = @as(f64, @floatFromInt(tokens_per_batch));
                        const tokens_per_sec = if (elapsed_s == 0) 0 else (steps_f * batch_tokens_f) / elapsed_s;
                        const partial_loss = batch_loss_sum / @as(f64, @floatFromInt(sample_idx + 1));
                        const display_loss = if (ema_initialized)
                            ema_loss * 0.95 + partial_loss * 0.05
                        else
                            partial_loss;
                        const label = std.fmt.bufPrint(
                            &progress_label_buf,
                            "loss {d:.3} tok/s {d:.0} step {d}/{d} ({d}/{d})",
                            .{
                                display_loss,
                                tokens_per_sec,
                                global_step,
                                total_steps,
                                sample_idx + 1,
                                batch_size,
                            },
                        ) catch "training";
                        progress_root.setName(label);
                        std.Progress.maybeRefresh();
                        last_label_update_ns = now_ns_sample;
                    }
                }
            }

            optimizer.step(&model, config.learning_rate);
            model.zeroGrad();

            const avg_batch_loss = batch_loss_sum / batch_size_f;
            epoch_loss_sum += avg_batch_loss;

            if (ema_initialized) {
                ema_loss = ema_loss * 0.95 + avg_batch_loss * 0.05;
            } else {
                ema_loss = avg_batch_loss;
                ema_initialized = true;
            }

            epoch_node.completeOne();
            progress_root.completeOne();

            const now_ns = timer.read();
            const since_last = if (last_label_update_ns == 0) now_ns else now_ns - last_label_update_ns;
            const update_label = last_label_update_ns == 0 or
                (since_last >= label_update_period_ns) or
                (epoch_steps == steps_per_epoch);

            if (update_label) {
                const elapsed_ns_f = @as(f64, @floatFromInt(now_ns));
                if (elapsed_ns_f > 0) {
                    const steps_f = @as(f64, @floatFromInt(global_step));
                    const batch_tokens_f = @as(f64, @floatFromInt(tokens_per_batch));
                    const tokens_per_sec = steps_f * batch_tokens_f * ns_per_s_f / elapsed_ns_f;
                    const elapsed_s = elapsed_ns_f / ns_per_s_f;
                    const secs_per_step = if (steps_f == 0 or elapsed_s == 0) 0 else elapsed_s / steps_f;
                    const epoch_remaining = if (epoch_steps >= steps_per_epoch) 0 else steps_per_epoch - epoch_steps;
                    const total_remaining = if (global_step >= total_steps) 0 else total_steps - global_step;
                    const eta_epoch = secs_per_step * @as(f64, @floatFromInt(epoch_remaining));
                    const eta_total = secs_per_step * @as(f64, @floatFromInt(total_remaining));

                    var elapsed_buf: [12]u8 = undefined;
                    var eta_epoch_buf: [12]u8 = undefined;
                    var eta_total_buf: [12]u8 = undefined;
                    var tok_buf: [12]u8 = undefined;
                    const elapsed_str = formatCompactDuration(&elapsed_buf, elapsed_s);
                    const eta_epoch_str = formatCompactDuration(&eta_epoch_buf, eta_epoch);
                    const eta_total_str = formatCompactDuration(&eta_total_buf, eta_total);
                    const tok_str = formatShortQuantity(&tok_buf, tokens_per_sec);

                    const label = std.fmt.bufPrint(
                        &progress_label_buf,
                        "L:{d:.3} tok:{s} t:{s} e:{s} T:{s}",
                        .{ ema_loss, tok_str, elapsed_str, eta_epoch_str, eta_total_str },
                    ) catch "training";
                    progress_root.setName(label);
                    std.Progress.maybeRefresh();
                    last_label_update_ns = now_ns;
                }
            }

            const should_log = (global_step % steps_log_interval == 0) or
                (epoch_steps == steps_per_epoch and config.epochs <= 5);
            if (should_log) {
                const elapsed_ns_f = @as(f64, @floatFromInt(now_ns));
                const elapsed_s = elapsed_ns_f / ns_per_s_f;
                const steps_f = @as(f64, @floatFromInt(global_step));
                const batch_tokens_f = @as(f64, @floatFromInt(tokens_per_batch));
                const tokens_per_sec = if (elapsed_s == 0) 0 else (steps_f * batch_tokens_f) / elapsed_s;
                const secs_per_step = if (steps_f == 0 or elapsed_s == 0) 0 else elapsed_s / steps_f;
                const epoch_remaining = if (epoch_steps >= steps_per_epoch) 0 else steps_per_epoch - epoch_steps;
                const total_remaining = if (global_step >= total_steps) 0 else total_steps - global_step;
                const eta_epoch = secs_per_step * @as(f64, @floatFromInt(epoch_remaining));
                const eta_total = secs_per_step * @as(f64, @floatFromInt(total_remaining));
                var eta_epoch_buf: [12]u8 = undefined;
                var eta_total_buf: [12]u8 = undefined;
                var tok_buf: [12]u8 = undefined;
                const eta_epoch_str = formatCompactDuration(&eta_epoch_buf, eta_epoch);
                const eta_total_str = formatCompactDuration(&eta_total_buf, eta_total);
                const tok_str = formatShortQuantity(&tok_buf, tokens_per_sec);
                std.log.info(
                    "step {d}/{d} (epoch {d}/{d}): ema loss {d:.4}, tok/s {s}, eta epoch {s}, eta total {s}",
                    .{
                        global_step,
                        total_steps,
                        epoch_idx + 1,
                        config.epochs,
                        ema_loss,
                        tok_str,
                        eta_epoch_str,
                        eta_total_str,
                    },
                );
            }
        }

        epoch_node.end();

        if (epoch_steps == 0) {
            std.log.warn("epoch {d} completed with zero steps", .{epoch_idx + 1});
            continue;
        }

        const epoch_elapsed_ns = timer.read() - epoch_start_elapsed;
        const epoch_elapsed_ns_f = @as(f64, @floatFromInt(epoch_elapsed_ns));
        const epoch_elapsed_s = epoch_elapsed_ns_f / ns_per_s_f;
        const epoch_tokens = epoch_steps * tokens_per_batch;
        const epoch_tokens_f = @as(f64, @floatFromInt(epoch_tokens));
        const epoch_tok_per_sec = if (epoch_elapsed_s == 0) 0 else epoch_tokens_f / epoch_elapsed_s;
        const epoch_avg_loss = epoch_loss_sum / @as(f64, @floatFromInt(epoch_steps));
        const steps_f = @as(f64, @floatFromInt(global_step));
        const total_elapsed_f = @as(f64, @floatFromInt(timer.read()));
        const total_elapsed_s = total_elapsed_f / ns_per_s_f;
        const secs_per_step = if (steps_f == 0 or total_elapsed_s == 0) 0 else total_elapsed_s / steps_f;
        const total_remaining_after_epoch = if (global_step >= total_steps) 0 else total_steps - global_step;
        const eta_total = secs_per_step * @as(f64, @floatFromInt(total_remaining_after_epoch));
        var eta_total_buf: [12]u8 = undefined;
        const eta_total_str = formatCompactDuration(&eta_total_buf, eta_total);
        var tok_buf: [12]u8 = undefined;
        const tok_str = formatShortQuantity(&tok_buf, epoch_tok_per_sec);

        std.log.info(
            "epoch {d}/{d}: avg loss {d:.4}, steps {d}, tok/s {s}, duration {d:.1}s, eta total {s}",
            .{
                epoch_idx + 1,
                config.epochs,
                epoch_avg_loss,
                epoch_steps,
                tok_str,
                epoch_elapsed_s,
                eta_total_str,
            },
        );

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

    const total_elapsed_ns = timer.read();
    const total_elapsed_ns_f = @as(f64, @floatFromInt(total_elapsed_ns));
    const total_elapsed_s = total_elapsed_ns_f / ns_per_s_f;
    const total_tokens = @as(f64, @floatFromInt(global_step)) *
        @as(f64, @floatFromInt(tokens_per_batch));
    const overall_tok_per_sec = if (total_elapsed_s == 0) 0 else total_tokens / total_elapsed_s;
    var elapsed_buf: [12]u8 = undefined;
    const elapsed_str = formatCompactDuration(&elapsed_buf, total_elapsed_s);
    var tok_buf: [12]u8 = undefined;
    const tok_str = formatShortQuantity(&tok_buf, overall_tok_per_sec);
    var zero_buf: [4]u8 = undefined;
    const zero_str = formatCompactDuration(&zero_buf, 0);

    if (ema_initialized and global_step > 0) {
        const final_label = std.fmt.bufPrint(
            &progress_label_buf,
            "L:{d:.3} tok:{s} t:{s} e:{s} T:{s}",
            .{ ema_loss, tok_str, elapsed_str, zero_str, zero_str },
        ) catch "complete";
        progress_root.setName(final_label);
    }
    std.Progress.setStatus(.success);

    std.log.info(
        "Training complete: {d} steps, duration {s}, tok/s {s}",
        .{ global_step, elapsed_str, tok_str },
    );
}

fn formatCheckpointName(
    allocator: std.mem.Allocator,
    dir: []const u8,
    prefix: []const u8,
    epoch: usize,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}_epoch{d}.ckpt", .{ dir, prefix, epoch });
}

fn formatCompactDuration(buf: []u8, seconds: f64) []const u8 {
    const clamped = if (seconds < 0) 0 else seconds;
    const total_secs = @as(u64, @intFromFloat(clamped));
    if (total_secs >= 3600) {
        const hours = total_secs / 3600;
        const minutes = (total_secs % 3600) / 60;
        return std.fmt.bufPrint(buf, "{d}h{d}m", .{ hours, minutes }) catch "–";
    } else if (total_secs >= 60) {
        const minutes = total_secs / 60;
        const secs = total_secs % 60;
        return std.fmt.bufPrint(buf, "{d}m{d}s", .{ minutes, secs }) catch "–";
    } else {
        return std.fmt.bufPrint(buf, "{d}s", .{total_secs}) catch "–";
    }
}

fn formatShortQuantity(buf: []u8, value: f64) []const u8 {
    const negative = value < 0;
    const abs_value = if (negative) -value else value;
    const prefix = if (negative) "-" else "";

    if (abs_value >= 1_000_000_000.0) {
        return std.fmt.bufPrint(buf, "{s}{d:.1}G", .{ prefix, abs_value / 1_000_000_000.0 }) catch "–";
    } else if (abs_value >= 1_000_000.0) {
        return std.fmt.bufPrint(buf, "{s}{d:.1}M", .{ prefix, abs_value / 1_000_000.0 }) catch "–";
    } else if (abs_value >= 1_000.0) {
        return std.fmt.bufPrint(buf, "{s}{d:.1}k", .{ prefix, abs_value / 1_000.0 }) catch "–";
    } else if (abs_value >= 100.0) {
        return std.fmt.bufPrint(buf, "{s}{d:.0}", .{ prefix, abs_value }) catch "–";
    } else {
        return std.fmt.bufPrint(buf, "{s}{d:.1}", .{ prefix, abs_value }) catch "–";
    }
}
