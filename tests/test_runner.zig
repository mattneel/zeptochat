const std = @import("std");
const builtin = @import("builtin");

pub fn main() void {
    const test_fns = builtin.test_functions;
    var passed: usize = 0;
    var failed: usize = 0;

    for (test_fns, 0..) |test_fn, idx| {
        std.debug.print("{d}/{d} {s}...", .{ idx + 1, test_fns.len, test_fn.name });
        test_fn.func() catch |err| {
            std.debug.print(" FAIL ({s})\n", .{@errorName(err)});
            failed += 1;
            continue;
        };
        std.debug.print(" OK\n", .{});
        passed += 1;
    }

    std.debug.print("\n{d} passed; {d} failed\n", .{ passed, failed });
    if (failed > 0) std.process.exit(1);
}

pub const std_options: std.Options = .{
    .logFn = customLog,
};

fn customLog(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = level;
    _ = scope;
    std.debug.print(format, args);
}
