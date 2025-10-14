const std = @import("std");
const transformer = @import("transformer");

pub const Transformer = transformer.Transformer;

pub fn crossEntropyLoss(
    logits: []const f32,
    targets: []const u32,
    seq_len: usize,
    vocab_size: usize,
) f32 {
    var total_loss: f32 = 0;

    for (0..seq_len) |i| {
        const logit_offset = i * vocab_size;
        const target = targets[i];

        var max_logit: f32 = -std.math.inf(f32);
        for (0..vocab_size) |j| {
            max_logit = @max(max_logit, logits[logit_offset + j]);
        }

        var sum_exp: f32 = 0;
        for (0..vocab_size) |j| {
            sum_exp += @exp(logits[logit_offset + j] - max_logit);
        }
        const log_sum_exp = @log(sum_exp);

        const target_logit = logits[logit_offset + target];
        total_loss -= (target_logit - max_logit - log_sum_exp);
    }

    return total_loss / @as(f32, @floatFromInt(seq_len));
}

pub fn crossEntropyGrad(
    allocator: std.mem.Allocator,
    logits: []const f32,
    targets: []const u32,
    seq_len: usize,
    vocab_size: usize,
) ![]f32 {
    var grad = try allocator.alloc(f32, logits.len);

    for (0..seq_len) |i| {
        const logit_offset = i * vocab_size;
        const target = targets[i];

        var max_logit: f32 = -std.math.inf(f32);
        for (0..vocab_size) |j| {
            max_logit = @max(max_logit, logits[logit_offset + j]);
        }

        var sum_exp: f32 = 0;
        for (0..vocab_size) |j| {
            sum_exp += @exp(logits[logit_offset + j] - max_logit);
        }

        const inv_seq_len = 1.0 / @as(f32, @floatFromInt(seq_len));
        for (0..vocab_size) |j| {
            const softmax = @exp(logits[logit_offset + j] - max_logit) / sum_exp;
            const one_hot: f32 = if (j == target) 1.0 else 0.0;
            grad[logit_offset + j] = (softmax - one_hot) * inv_seq_len;
        }
    }

    return grad;
}

pub fn sgdStep(model: *Transformer, learning_rate: f32) void {
    const lr = learning_rate;

    for (model.token_embeddings, model.grad_token_embeddings) |*param, grad| {
        param.* -= lr * grad;
    }

    for (model.position_embeddings, model.grad_position_embeddings) |*param, grad| {
        param.* -= lr * grad;
    }

    for (model.lm_head, model.grad_lm_head) |*param, grad| {
        param.* -= lr * grad;
    }

    for (model.layers) |*layer| {
        for (layer.ln1.weight, layer.ln1.grad_weight) |*w, g| w.* -= lr * g;
        for (layer.ln1.bias, layer.ln1.grad_bias) |*b, g| b.* -= lr * g;

        for (layer.attn.w_q, layer.attn.grad_w_q) |*w, g| w.* -= lr * g;
        for (layer.attn.w_k, layer.attn.grad_w_k) |*w, g| w.* -= lr * g;
        for (layer.attn.w_v, layer.attn.grad_w_v) |*w, g| w.* -= lr * g;
        for (layer.attn.w_o, layer.attn.grad_w_o) |*w, g| w.* -= lr * g;

        for (layer.ln2.weight, layer.ln2.grad_weight) |*w, g| w.* -= lr * g;
        for (layer.ln2.bias, layer.ln2.grad_bias) |*b, g| b.* -= lr * g;

        for (layer.mlp.w1, layer.mlp.grad_w1) |*w, g| w.* -= lr * g;
        for (layer.mlp.b1, layer.mlp.grad_b1) |*b, g| b.* -= lr * g;
        for (layer.mlp.w2, layer.mlp.grad_w2) |*w, g| w.* -= lr * g;
        for (layer.mlp.b2, layer.mlp.grad_b2) |*b, g| b.* -= lr * g;
    }

    for (model.ln_final.weight, model.ln_final.grad_weight) |*w, g| w.* -= lr * g;
    for (model.ln_final.bias, model.ln_final.grad_bias) |*b, g| b.* -= lr * g;
}
