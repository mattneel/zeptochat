const std = @import("std");
const Tokenizer = @import("tokenizer").Tokenizer;

test "tokenizer encodes ASCII fallback" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.init(allocator);
    defer tokenizer.deinit();

    const input = "hi";
    const tokens = try tokenizer.encode(input);
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, input.len), tokens.len);
    try std.testing.expectEqual(@as(u32, 'h'), tokens[0]);
    try std.testing.expectEqual(@as(u32, 'i'), tokens[1]);

    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(input, decoded);
}

test "tokenizer merges simple pair" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.init(allocator);
    defer tokenizer.deinit();

    try tokenizer.addToken("hi", 256);
    try tokenizer.addMerge(.{
        .left = "h",
        .right = "i",
        .result = "hi",
        .rank = 0,
    });

    const tokens = try tokenizer.encode("hi");
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 256), tokens[0]);

    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("hi", decoded);
}
