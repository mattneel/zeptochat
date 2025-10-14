const std = @import("std");
const Tokenizer = @import("tokenizer").Tokenizer;

test "tokenizer encodes ASCII fallback" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
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

test "tokenizer falls back to byte tokens for utf8" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
    defer tokenizer.deinit();

    const input = "🤖";
    const tokens = try tokenizer.encode(input);
    defer allocator.free(tokens);

    const expected_bytes = input;
    try std.testing.expectEqual(@as(usize, expected_bytes.len), tokens.len);
    for (tokens, 0..) |token, idx| {
        try std.testing.expectEqual(@as(u32, expected_bytes[idx]), token);
    }

    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(input, decoded);
}

test "tokenizer merges simple pair" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
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

test "tokenizer prefers lowest-rank merge" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
    defer tokenizer.deinit();

    try tokenizer.addToken("ab", 300);
    try tokenizer.addToken("ba", 301);

    try tokenizer.addMerge(.{
        .left = "a",
        .right = "b",
        .result = "ab",
        .rank = 5,
    });

    try tokenizer.addMerge(.{
        .left = "b",
        .right = "a",
        .result = "ba",
        .rank = 1,
    });

    const tokens = try tokenizer.encode("abab");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 3), tokens.len);
    try std.testing.expectEqual(@as(u32, 'a'), tokens[0]);
    try std.testing.expectEqual(@as(u32, 301), tokens[1]);
    try std.testing.expectEqual(@as(u32, 'b'), tokens[2]);
}

test "tokenizer collapses cascading merges" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
    defer tokenizer.deinit();

    try tokenizer.addToken("aa", 600);
    try tokenizer.addToken("aaaa", 601);

    try tokenizer.addMerge(.{
        .left = "a",
        .right = "a",
        .result = "aa",
        .rank = 0,
    });

    try tokenizer.addMerge(.{
        .left = "aa",
        .right = "aa",
        .result = "aaaa",
        .rank = 1,
    });

    const tokens = try tokenizer.encode("aaaa");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 601), tokens[0]);
}

test "tokenizer rejects merges with unknown symbols" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
    defer tokenizer.deinit();

    try tokenizer.addToken("ok", 512);

    const merge_err = tokenizer.addMerge(.{
        .left = "x",
        .right = "y",
        .result = "xy",
        .rank = 0,
    });
    try std.testing.expectError(error.UnknownToken, merge_err);
}

test "tokenizer loads vocab and merges from buffers" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
    defer tokenizer.deinit();

    const vocab_bytes =
        \\256 hi
        \\104 h
        \\105 i
    ;

    const merge_bytes =
        \\h i hi
    ;

    try tokenizer.loadVocabFromBytes(vocab_bytes);
    try tokenizer.loadMergesFromBytes(merge_bytes);

    const tokens = try tokenizer.encode("hi");
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 256), tokens[0]);
}

test "tokenizer reports vocab size including merges" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.initEmpty(allocator);
    defer tokenizer.deinit();

    try std.testing.expectEqual(@as(usize, 256), tokenizer.vocabSize());

    try tokenizer.addToken("hi", 512);
    try std.testing.expectEqual(@as(usize, 257), tokenizer.vocabSize());
}

test "tokenizer init from files" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var vocab_file = try tmp.dir.createFile("vocab.txt", .{});
    defer vocab_file.close();
    var vocab_buffer: [256]u8 = undefined;
    var vocab_writer = vocab_file.writer(&vocab_buffer);
    const vocab_io: *std.Io.Writer = &vocab_writer.interface;
    const vocab: []const u8 =
        \\256 hi
        \\104 h
        \\105 i
    ;
    try vocab_io.print("{s}", .{vocab});
    try vocab_io.flush();

    var merges_file = try tmp.dir.createFile("merges.txt", .{});
    defer merges_file.close();
    var merges_buffer: [256]u8 = undefined;
    var merges_writer = merges_file.writer(&merges_buffer);
    const merges_io: *std.Io.Writer = &merges_writer.interface;
    const merges: []const u8 = "h i hi\n";
    try merges_io.print("{s}", .{merges});
    try merges_io.flush();

    var tokenizer = try Tokenizer.initFromDir(allocator, tmp.dir, "vocab.txt", "merges.txt");
    defer tokenizer.deinit();

    const tokens = try tokenizer.encode("hi");
    defer allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 256), tokens[0]);
}

test "tokenizer gpt2 fixture known tokens" {
    const allocator = std.testing.allocator;
    const cwd = std.fs.cwd();

    var tokenizer = try Tokenizer.initFromDir(
        allocator,
        cwd,
        "tests/fixtures/gpt2_mini/vocab.txt",
        "tests/fixtures/gpt2_mini/merges.txt",
    );
    defer tokenizer.deinit();

    try std.testing.expectEqual(@as(u32, 262), tokenizer.lookupTokenId("Ġthe").?);
    try std.testing.expectEqual(@as(u32, 290), tokenizer.lookupTokenId("Ġand").?);
    try std.testing.expectEqual(@as(u32, 995), tokenizer.lookupTokenId("Ġworld").?);
}
