const std = @import("std");

pub fn main() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    try stdout.print(
        \\Zeptochat benchmarks are not implemented yet.
        \\Implement matmul and attention microbenchmarks, then update this entry point.
        \\See TODO.md · Phase 2 · SIMD.
        \\
    , .{});
    try stdout.flush();
}
