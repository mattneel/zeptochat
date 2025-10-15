const std = @import("std");
const testing = std.testing;
const dataset_mod = @import("dataset");

const Dataset = dataset_mod.Dataset;

test "dataset iterator yields contiguous batches" {
    const tokens = [_]u32{ 10, 11, 12, 13, 14, 15, 16 };
    var ds = try Dataset.initFromSlice(testing.allocator, &tokens);
    defer ds.deinit();

    var iter = ds.iterator(2, 3); // batch_size=2, seq_len=3 -> tokens=6
    const batch = iter.next() orelse unreachable;

    try testing.expectEqual(@as(usize, 6), batch.tokens.len);
    try testing.expectEqual(@as(usize, 6), batch.targets.len);

    try testing.expectEqual(@as(u32, 10), batch.tokens[0]);
    try testing.expectEqual(@as(u32, 11), batch.targets[0]);

    try testing.expectEqual(@as(u32, 15), batch.tokens[5]);
    try testing.expectEqual(@as(u32, 16), batch.targets[5]);

    try testing.expect(iter.next() == null);
}

test "dataset iterator handles reset and remaining" {
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var ds = try Dataset.initFromSlice(testing.allocator, &tokens);
    defer ds.deinit();

    var iter = ds.iterator(1, 3);
    try testing.expectEqual(@as(usize, 2), iter.remaining()); // floor((9-0-1)/3)

    _ = iter.next() orelse unreachable;
    try testing.expectEqual(@as(usize, 1), iter.remaining());

    iter.reset();
    try testing.expectEqual(@as(usize, 2), iter.remaining());
}
