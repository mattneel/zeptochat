const std = @import("std");

pub const ModelConfig = struct {
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    dropout: f32,
};

/// Minimal configuration for rapid iteration and unit tests.
pub const TinyConfig = ModelConfig{
    .vocab_size = 256,
    .context_length = 32,
    .d_model = 64,
    .n_heads = 2,
    .n_layers = 2,
    .dropout = 0.0,
};

pub const Transformer = struct {
    config: ModelConfig,
    allocator: std.mem.Allocator,
    token_embeddings: []f32,
    position_embeddings: []f32,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !Transformer {
        const token_embeddings = try allocator.alloc(f32, config.vocab_size * config.d_model);
        errdefer allocator.free(token_embeddings);

        const position_embeddings = try allocator.alloc(f32, config.context_length * config.d_model);
        errdefer allocator.free(position_embeddings);

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        const scale = @sqrt(1.0 / @as(f32, @floatFromInt(config.d_model)));

        for (token_embeddings) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }

        for (position_embeddings) |*val| {
            val.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }

        return Transformer{
            .config = config,
            .allocator = allocator,
            .token_embeddings = token_embeddings,
            .position_embeddings = position_embeddings,
        };
    }

    pub fn deinit(self: *Transformer) void {
        self.allocator.free(self.token_embeddings);
        self.allocator.free(self.position_embeddings);
    }

    pub fn forward(self: *Transformer, tokens: []const u32) ![]f32 {
        const batch_size = tokens.len;
        const d_model = self.config.d_model;

        var hidden = try self.allocator.alloc(f32, batch_size * d_model);
        errdefer self.allocator.free(hidden);

        for (tokens, 0..) |token, pos| {
            const token_offset = token * d_model;
            const pos_offset = pos * d_model;
            const hidden_offset = pos * d_model;

            for (0..d_model) |i| {
                hidden[hidden_offset + i] =
                    self.token_embeddings[token_offset + i] +
                    self.position_embeddings[pos_offset + i];
            }
        }

        return hidden;
    }
};
