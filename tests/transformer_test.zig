const std = @import("std");
const testing = std.testing;
const Transformer = @import("transformer").Transformer;
const TinyConfig = @import("transformer").TinyConfig;

test "transformer init allocates embeddings" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    try testing.expectEqual(TinyConfig.vocab_size * TinyConfig.d_model, model.token_embeddings.len);
    try testing.expectEqual(TinyConfig.context_length * TinyConfig.d_model, model.position_embeddings.len);
}

test "transformer forward returns correct shape" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const tokens = [_]u32{ 1, 2, 3, 4 };
    const hidden = try model.forward(&tokens);
    defer testing.allocator.free(hidden);

    try testing.expectEqual(tokens.len * TinyConfig.d_model, hidden.len);
}

test "transformer embeddings are randomized" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    const d_model = TinyConfig.d_model;
    const token0 = model.token_embeddings[0..d_model];
    const token1 = model.token_embeddings[d_model .. d_model * 2];

    var identical = true;
    for (token0, token1) |a, b| {
        if (a != b) {
            identical = false;
            break;
        }
    }

    try testing.expect(!identical);
}
