const std = @import("std");
const testing = std.testing;
const parallel = @import("parallel");

test "threadpool: basic submit and wait" {
    var pool = try parallel.ThreadPool.init(testing.allocator, 4);
    defer pool.deinit();

    var counter = std.atomic.Value(usize).init(0);

    const Job = struct {
        fn run(value: *std.atomic.Value(usize)) void {
            _ = value.fetchAdd(1, .monotonic);
        }
    };

    for (0..100) |_| {
        try pool.submit(std.atomic.Value(usize), Job.run, &counter);
    }

    pool.wait();

    try testing.expectEqual(@as(usize, 100), counter.load(.monotonic));
}

test "threadpool: parallelFor" {
    var pool = try parallel.ThreadPool.init(testing.allocator, 4);
    defer pool.deinit();

    const results = try testing.allocator.alloc(usize, 512);
    defer testing.allocator.free(results);
    @memset(results, 0);

    const Kernel = struct {
        var data: []usize = undefined;

        fn compute(i: usize) void {
            data[i] = i * 2;
        }
    };

    Kernel.data = results;
    try pool.parallelFor(Kernel.compute, 0, results.len);

    for (results, 0..) |val, idx| {
        try testing.expectEqual(idx * 2, val);
    }
}

test "threadpool: parallelMap" {
    var pool = try parallel.ThreadPool.init(testing.allocator, 4);
    defer pool.deinit();

    const input = try testing.allocator.alloc(i32, 128);
    defer testing.allocator.free(input);
    const output = try testing.allocator.alloc(i32, 128);
    defer testing.allocator.free(output);

    for (input, 0..) |*val, idx| {
        val.* = @intCast(idx);
    }

    const square = struct {
        fn f(x: i32) i32 {
            return x * x;
        }
    }.f;

    try pool.parallelMap(i32, i32, square, input, output);

    for (output, 0..) |val, idx| {
        try testing.expectEqual(@as(i32, @intCast(idx * idx)), val);
    }
}

test "parallelRange helper" {
    const results = try testing.allocator.alloc(usize, 256);
    defer testing.allocator.free(results);
    @memset(results, 0);

    const Kernel = struct {
        var data: []usize = undefined;

        fn compute(i: usize) void {
            data[i] = i * 3;
        }
    };

    Kernel.data = results;
    try parallel.parallelRange(
        testing.allocator,
        Kernel.compute,
        0,
        results.len,
        4,
    );

    for (results, 0..) |val, idx| {
        try testing.expectEqual(idx * 3, val);
    }
}
