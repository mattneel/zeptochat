const std = @import("std");
const testing = std.testing;
const transformer = @import("transformer");
const checkpoint = @import("checkpoint");

const Transformer = transformer.Transformer;
const TinyConfig = transformer.TinyConfig;

test "checkpoint save/load round trip" {
    var model = try Transformer.init(testing.allocator, TinyConfig);
    defer model.deinit();

    // Stamp deterministic values so we can compare later.
    for (model.token_embeddings, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i % 17)) * 0.01;
    }
    if (model.token_embeddings.len > 0) model.token_embeddings[0] = 0.42;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file_name = try std.fmt.allocPrint(
        testing.allocator,
        ".zig-cache/tmp/{s}/checkpoint.bin",
        .{tmp.sub_path[0..]},
    );
    defer testing.allocator.free(file_name);

    try checkpoint.save(&model, file_name, 123);

    var loaded = try checkpoint.load(testing.allocator, file_name);
    defer loaded.model.deinit();

    try testing.expectEqual(@as(usize, 123), loaded.meta.step);
    try testing.expectEqual(loaded.meta.config.vocab_size, model.config.vocab_size);
    try testing.expectEqual(loaded.meta.config.context_length, model.config.context_length);

    try testing.expectEqualSlices(f32, model.token_embeddings, loaded.model.token_embeddings);
    try testing.expectEqualSlices(f32, model.position_embeddings, loaded.model.position_embeddings);
}
