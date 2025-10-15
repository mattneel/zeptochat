const std = @import("std");
const transformer = @import("transformer");

const Transformer = transformer.Transformer;

pub const AdamW = struct {
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    m: []f32,
    v: []f32,
    step_count: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_params: usize) !AdamW {
        const m = try allocator.alloc(f32, num_params);
        errdefer allocator.free(m);

        const v = try allocator.alloc(f32, num_params);
        errdefer allocator.free(v);

        @memset(m, 0.0);
        @memset(v, 0.0);

        return AdamW{
            .m = m,
            .v = v,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AdamW) void {
        self.allocator.free(self.m);
        self.allocator.free(self.v);
        self.* = undefined;
    }

    pub fn countParams(model: *const Transformer) usize {
        var count: usize = 0;
        count += model.token_embeddings.len;
        count += model.position_embeddings.len;
        count += model.lm_head.len;

        for (model.layers) |layer| {
            count += layer.ln1.weight.len + layer.ln1.bias.len;
            count += layer.attn.w_q.len + layer.attn.w_k.len + layer.attn.w_v.len + layer.attn.w_o.len;
            count += layer.ln2.weight.len + layer.ln2.bias.len;
            count += layer.mlp.w1.len + layer.mlp.b1.len + layer.mlp.w2.len + layer.mlp.b2.len;
        }

        count += model.ln_final.weight.len + model.ln_final.bias.len;
        return count;
    }

    pub fn step(self: *AdamW, model: *Transformer, learning_rate: f32) void {
        self.step_count += 1;
        const step_f = @as(f32, @floatFromInt(self.step_count));
        const bias_correction1 = 1.0 - std.math.pow(f32, self.beta1, step_f);
        const bias_correction2 = 1.0 - std.math.pow(f32, self.beta2, step_f);
        const adjusted_lr = learning_rate * @sqrt(bias_correction2) / bias_correction1;

        var offset: usize = 0;
        offset = self.updateParams(model.token_embeddings, model.grad_token_embeddings, offset, adjusted_lr);
        offset = self.updateParams(model.position_embeddings, model.grad_position_embeddings, offset, adjusted_lr);
        offset = self.updateParams(model.lm_head, model.grad_lm_head, offset, adjusted_lr);

        for (model.layers) |*layer| {
            offset = self.updateParams(layer.ln1.weight, layer.ln1.grad_weight, offset, adjusted_lr);
            offset = self.updateParams(layer.ln1.bias, layer.ln1.grad_bias, offset, adjusted_lr);
            offset = self.updateParams(layer.attn.w_q, layer.attn.grad_w_q, offset, adjusted_lr);
            offset = self.updateParams(layer.attn.w_k, layer.attn.grad_w_k, offset, adjusted_lr);
            offset = self.updateParams(layer.attn.w_v, layer.attn.grad_w_v, offset, adjusted_lr);
            offset = self.updateParams(layer.attn.w_o, layer.attn.grad_w_o, offset, adjusted_lr);
            offset = self.updateParams(layer.ln2.weight, layer.ln2.grad_weight, offset, adjusted_lr);
            offset = self.updateParams(layer.ln2.bias, layer.ln2.grad_bias, offset, adjusted_lr);
            offset = self.updateParams(layer.mlp.w1, layer.mlp.grad_w1, offset, adjusted_lr);
            offset = self.updateParams(layer.mlp.b1, layer.mlp.grad_b1, offset, adjusted_lr);
            offset = self.updateParams(layer.mlp.w2, layer.mlp.grad_w2, offset, adjusted_lr);
            offset = self.updateParams(layer.mlp.b2, layer.mlp.grad_b2, offset, adjusted_lr);
        }

        offset = self.updateParams(model.ln_final.weight, model.ln_final.grad_weight, offset, adjusted_lr);
        _ = self.updateParams(model.ln_final.bias, model.ln_final.grad_bias, offset, adjusted_lr);
    }

    fn updateParams(
        self: *AdamW,
        params: []f32,
        grads: []const f32,
        offset: usize,
        lr: f32,
    ) usize {
        const end = offset + params.len;
        std.debug.assert(end <= self.m.len and end <= self.v.len);

        for (params, grads, 0..) |*param, grad, i| {
            const idx = offset + i;

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * grad;
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * grad * grad;

            param.* *= (1.0 - lr * self.weight_decay);
            param.* -= lr * self.m[idx] / (@sqrt(self.v[idx]) + self.eps);
        }

        return end;
    }
};
