const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    const exe_name = args.next().?;
    const output_path = args.next() orelse {
        std.debug.print("Usage: {s} <output_path> [repetitions]\n", .{exe_name});
        return ToyDataError.MissingOutputPath;
    };

    const repetitions_arg = args.next();
    const repetitions = repetitions_arg orelse "1000";
    const repeat_count = std.fmt.parseInt(usize, repetitions, 10) catch {
        std.debug.print("error: invalid repetition count '{s}'\n", .{repetitions});
        return ToyDataError.InvalidRepetitionCount;
    };

    if (repeat_count == 0) return ToyDataError.InvalidRepetitionCount;

    const pattern = [_]u32{ 1, 2, 3, 4, 5 };
    const total_tokens = pattern.len * repeat_count;

    var tokens = try allocator.alloc(u32, total_tokens);
    defer allocator.free(tokens);

    for (0..repeat_count) |rep| {
        const base = rep * pattern.len;
        for (pattern, 0..) |value, idx| {
            tokens[base + idx] = value;
        }
    }

    const directory = std.fs.cwd();
    try directory.makePath(std.fs.path.dirname(output_path) orelse ".");

    var file = try directory.createFile(output_path, .{ .truncate = true });
    defer file.close();

    try file.writeAll(std.mem.sliceAsBytes(tokens));

    std.debug.print(
        "Wrote {d} tokens to {s} using pattern [1,2,3,4,5] repeated {d} times\n",
        .{ tokens.len, output_path, repeat_count },
    );
}

const ToyDataError = error{ MissingOutputPath, InvalidRepetitionCount };
