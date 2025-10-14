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
