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

const Worker = struct {
    model: Transformer,
    loss_sum: f64 = 0,
};

const WorkerContext = struct {
    worker: *Worker,
    batch_tokens: []const u32,
    batch_targets: []const u32,
    seq_len: usize,
    vocab_size: usize,
    inv_batch_f32: f32,
    start_index: usize,
    end_index: usize,
    completed_samples: *std.atomic.Value(usize),
    loss_out: *f64,
    err: ?anyerror = null,
};

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

    var model_config = config.model;
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

    var max_token: u32 = 0;
    for (dataset.tokens) |tok| {
        if (tok > max_token) max_token = tok;
    }
    const required_vocab = @as(usize, @intCast(max_token)) + 1;
    if (required_vocab > model_config.vocab_size) {
        std.log.warn(
            "expanding vocab size from {d} to {d} to cover dataset tokens",
            .{ model_config.vocab_size, required_vocab },
        );
        model_config.vocab_size = required_vocab;
    }

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

    const max_workers = @max(@as(usize, 1), thread_pool.threadCount());
    var workers = try allocator.alloc(Worker, max_workers);
    defer {
        for (workers) |*worker| worker.model.deinit();
        allocator.free(workers);
    }
    for (workers) |*worker| {
        worker.model = try Transformer.init(allocator, model_config);
        copyTransformerParameters(&worker.model, &model);
    }

    var worker_contexts = try allocator.alloc(WorkerContext, max_workers);
    defer allocator.free(worker_contexts);

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
    progress_root.setName("L:---- tok:---- T:----");

    var progress_label_buf: [std.Progress.Node.max_name_len]u8 = undefined;
    var ema_loss: f64 = 0;
    var ema_initialized = false;
    var last_label_update_ns: u64 = 0;

    for (0..config.epochs) |epoch_idx| {
        const epoch_start_elapsed = timer.read();
        var iter = dataset.iterator(config.batch_size, seq_len);

        var epoch_node = progress_root.start("", steps_per_epoch);
        epoch_node.setEstimatedTotalItems(steps_per_epoch);
        epoch_node.setCompletedItems(0);
        var epoch_label_buf: [std.Progress.Node.max_name_len]u8 = undefined;
        const epoch_label = std.fmt.bufPrint(
            &epoch_label_buf,
            "ep {d}/{d} e:--",
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

            var completed_samples = std.atomic.Value(usize).init(0);

            var batch_node: ?std.Progress.Node = null;
            if (batch_size > 1) {
                batch_node = epoch_node.start("", batch_size);
                batch_node.?.setEstimatedTotalItems(batch_size);
                batch_node.?.setCompletedItems(0);
            }
            defer if (batch_node) |node| node.end();

            const active_workers = @min(max_workers, batch_size);
            const chunk_size = (batch_size + active_workers - 1) / active_workers;

            var job_count: usize = 0;
            var start_index: usize = 0;
            while (start_index < batch_size) {
                const end_index = @min(start_index + chunk_size, batch_size);
                var worker = &workers[job_count];
                copyTransformerParameters(&worker.model, &model);
                worker.model.zeroGrad();
                worker.loss_sum = 0;

                worker_contexts[job_count] = .{
                    .worker = worker,
                    .batch_tokens = batch.tokens,
                    .batch_targets = batch.targets,
                    .seq_len = seq_len_usize,
                    .vocab_size = vocab_size,
                    .inv_batch_f32 = inv_batch_f32,
                    .start_index = start_index,
                    .end_index = end_index,
                    .completed_samples = &completed_samples,
                    .loss_out = &worker.loss_sum,
                    .err = null,
                };
                const ctx = &worker_contexts[job_count];
                try thread_pool.submit(WorkerContext, workerJob, ctx);

                job_count += 1;
                start_index = end_index;
            }

            var observed_samples: usize = 0;
            while (true) {
                const done = completed_samples.load(.acquire);
                if (done > observed_samples) {
                    observed_samples = done;
                    updateBatchProgress(
                        &timer,
                        progress_root,
                        epoch_node,
                        batch_node,
                        done,
                        batch_size,
                        seq_len_usize,
                        global_step,
                        total_steps,
                        steps_per_epoch,
                        epoch_idx,
                        config.epochs,
                        tokens_per_batch,
                        ema_loss,
                        &progress_label_buf,
                        &epoch_label_buf,
                        &last_label_update_ns,
                    );
                }
                if (done >= batch_size) break;
                std.Thread.sleep(ns_per_s / 200);
            }

            thread_pool.wait();

            var batch_loss_sum: f64 = 0;
            var worker_index: usize = 0;
            while (worker_index < job_count) : (worker_index += 1) {
                const ctx = worker_contexts[worker_index];
                if (ctx.err) |err| return err;
                const worker = &workers[worker_index];
                batch_loss_sum += worker.loss_sum;
                accumulateTransformerGradients(&model, &worker.model);
                worker.model.zeroGrad();
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
            if (last_label_update_ns == 0 or since_last >= label_update_period_ns or epoch_steps == steps_per_epoch) {
                updateBatchProgress(
                    &timer,
                    progress_root,
                    epoch_node,
                    batch_node,
                    batch_size,
                    batch_size,
                    seq_len_usize,
                    global_step,
                    total_steps,
                    steps_per_epoch,
                    epoch_idx,
                    config.epochs,
                    tokens_per_batch,
                    ema_loss,
                    &progress_label_buf,
                    &epoch_label_buf,
                    &last_label_update_ns,
                );
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

fn workerJob(ctx: *WorkerContext) void {
    var loss_sum: f64 = 0;

    const worker = ctx.worker;
    const allocator = worker.model.allocator;
    const seq_len = ctx.seq_len;

    var sample_index = ctx.start_index;
    while (sample_index < ctx.end_index) : (sample_index += 1) {
        const offset = sample_index * seq_len;
        const sample_tokens = ctx.batch_tokens[offset .. offset + seq_len];
        const sample_targets = ctx.batch_targets[offset .. offset + seq_len];

        const logits = worker.model.forward(sample_tokens) catch |err| {
            ctx.err = err;
            return;
        };
        defer allocator.free(logits);

        const loss_f32 = training.crossEntropyLoss(logits, sample_targets, seq_len, ctx.vocab_size);
        loss_sum += @as(f64, loss_f32);

        const grad_logits = training.crossEntropyGrad(
            allocator,
            logits,
            sample_targets,
            ctx.seq_len,
            ctx.vocab_size,
        ) catch |err| {
            ctx.err = err;
            return;
        };
        defer allocator.free(grad_logits);

        for (grad_logits) |*g| g.* *= ctx.inv_batch_f32;

        worker.model.backward(grad_logits) catch |err| {
            ctx.err = err;
            return;
        };

        _ = ctx.completed_samples.fetchAdd(1, .acq_rel) + 1;
    }

    ctx.loss_out.* = loss_sum;
}

fn updateBatchProgress(
    timer: *std.time.Timer,
    root_node: std.Progress.Node,
    epoch_node: std.Progress.Node,
    batch_node: ?std.Progress.Node,
    samples_done: usize,
    batch_size: usize,
    seq_len: usize,
    global_step: usize,
    total_steps: usize,
    steps_per_epoch: usize,
    epoch_index: usize,
    total_epochs: usize,
    tokens_per_batch: usize,
    ema_loss: f64,
    root_label_buf: *[std.Progress.Node.max_name_len]u8,
    epoch_label_buf: *[std.Progress.Node.max_name_len]u8,
    last_label_update_ns: *u64,
) void {
    if (batch_node) |node| {
        var handle = node;
        handle.setCompletedItems(samples_done);
        handle.setName(std.fmt.bufPrint(
            epoch_label_buf,
            "s {d}/{d}",
            .{ samples_done, batch_size },
        ) catch "sample");
    }

    const now_ns = timer.read();
    const elapsed_ns_f = @as(f64, @floatFromInt(now_ns));
    if (elapsed_ns_f == 0) return;

    const elapsed_s = elapsed_ns_f / ns_per_s_f;
    const tokens_processed = @as(f64, @floatFromInt((global_step - 1) * tokens_per_batch + samples_done * seq_len));
    const tokens_per_sec = if (elapsed_s == 0) 0 else tokens_processed / elapsed_s;

    const partial_step = @as(f64, @floatFromInt(samples_done)) / @as(f64, @floatFromInt(batch_size));
    const steps_completed = @as(f64, @floatFromInt(global_step - 1)) + partial_step;
    const secs_per_step = if (steps_completed == 0 or elapsed_s == 0) 0 else elapsed_s / steps_completed;

    const epoch_steps_f = @as(f64, @floatFromInt(steps_per_epoch));
    const step_in_epoch = (@as(f64, @floatFromInt((global_step - 1) % steps_per_epoch))) + partial_step;
    const epoch_remaining = if (epoch_steps_f <= step_in_epoch) 0 else epoch_steps_f - step_in_epoch;
    const total_steps_f = @as(f64, @floatFromInt(total_steps));
    const total_remaining = if (total_steps_f <= steps_completed) 0 else total_steps_f - steps_completed;

    const eta_epoch = secs_per_step * epoch_remaining;
    const eta_total = secs_per_step * total_remaining;

    var buf_eta_epoch: [12]u8 = undefined;
    var buf_eta_total: [12]u8 = undefined;
    var buf_tok: [12]u8 = undefined;
    const eta_epoch_str = formatCompactDuration(&buf_eta_epoch, eta_epoch);
    const eta_total_str = formatCompactDuration(&buf_eta_total, eta_total);
    const tok_str = formatShortQuantity(&buf_tok, tokens_per_sec);

    const root_label = std.fmt.bufPrint(
        root_label_buf,
        "L:{d:.3} tok:{s} T:{s}",
        .{ ema_loss, tok_str, eta_total_str },
    ) catch "root";
    root_node.setName(root_label);

    const epoch_label = std.fmt.bufPrint(
        epoch_label_buf,
        "ep {d}/{d} e:{s}",
        .{ epoch_index + 1, total_epochs, eta_epoch_str },
    ) catch "epoch";
    epoch_node.setName(epoch_label);

    const base_steps: usize = (global_step - 1) % steps_per_epoch;
    const epoch_completed = base_steps + @intFromBool(samples_done == batch_size);
    epoch_node.setCompletedItems(epoch_completed);
    last_label_update_ns.* = now_ns;
}

fn copyTransformerParameters(dst: *Transformer, src: *const Transformer) void {
    copySlice(dst.token_embeddings, src.token_embeddings);
    copySlice(dst.position_embeddings, src.position_embeddings);
    copySlice(dst.lm_head, src.lm_head);
    copyLayerNormParams(&dst.ln_final, &src.ln_final);

    for (dst.layers, src.layers) |*dst_layer, *src_layer| {
        copyLayerNormParams(&dst_layer.ln1, &src_layer.ln1);
        copyAttentionParams(&dst_layer.attn, &src_layer.attn);
        copyLayerNormParams(&dst_layer.ln2, &src_layer.ln2);
        copyMlpParams(&dst_layer.mlp, &src_layer.mlp);
    }
}

fn accumulateTransformerGradients(dst: *Transformer, src: *const Transformer) void {
    addSlice(dst.grad_token_embeddings, src.grad_token_embeddings);
    addSlice(dst.grad_position_embeddings, src.grad_position_embeddings);
    addSlice(dst.grad_lm_head, src.grad_lm_head);
    addSlice(dst.ln_final.grad_weight, src.ln_final.grad_weight);
    addSlice(dst.ln_final.grad_bias, src.ln_final.grad_bias);

    for (dst.layers, src.layers) |*dst_layer, *src_layer| {
        addSlice(dst_layer.ln1.grad_weight, src_layer.ln1.grad_weight);
        addSlice(dst_layer.ln1.grad_bias, src_layer.ln1.grad_bias);
        accumulateAttentionGradients(&dst_layer.attn, &src_layer.attn);
        addSlice(dst_layer.ln2.grad_weight, src_layer.ln2.grad_weight);
        addSlice(dst_layer.ln2.grad_bias, src_layer.ln2.grad_bias);
        accumulateMlpGradients(&dst_layer.mlp, &src_layer.mlp);
    }
}

fn copySlice(dst: []f32, src: []const f32) void {
    @memcpy(dst, src);
}

fn addSlice(dst: []f32, src: []const f32) void {
    for (dst, src) |*d, s| d.* += s;
}

fn copyLayerNormParams(dst: *transformer.LayerNorm, src: *const transformer.LayerNorm) void {
    copySlice(dst.weight, src.weight);
    copySlice(dst.bias, src.bias);
}

fn copyAttentionParams(dst: *transformer.MultiHeadAttention, src: *const transformer.MultiHeadAttention) void {
    copySlice(dst.w_q, src.w_q);
    copySlice(dst.w_k, src.w_k);
    copySlice(dst.w_v, src.w_v);
    copySlice(dst.w_o, src.w_o);
}

fn copyMlpParams(dst: *transformer.MLP, src: *const transformer.MLP) void {
    copySlice(dst.w1, src.w1);
    copySlice(dst.b1, src.b1);
    copySlice(dst.w2, src.w2);
    copySlice(dst.b2, src.b2);
}

fn accumulateAttentionGradients(dst: *transformer.MultiHeadAttention, src: *const transformer.MultiHeadAttention) void {
    addSlice(dst.grad_w_q, src.grad_w_q);
    addSlice(dst.grad_w_k, src.grad_w_k);
    addSlice(dst.grad_w_v, src.grad_w_v);
    addSlice(dst.grad_w_o, src.grad_w_o);
}

fn accumulateMlpGradients(dst: *transformer.MLP, src: *const transformer.MLP) void {
    addSlice(dst.grad_w1, src.grad_w1);
    addSlice(dst.grad_b1, src.grad_b1);
    addSlice(dst.grad_w2, src.grad_w2);
    addSlice(dst.grad_b2, src.grad_b2);
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
