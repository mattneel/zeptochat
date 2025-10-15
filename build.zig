const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "zeptochat",
        .root_module = root_module,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run zeptochat CLI");
    run_step.dependOn(&run_cmd.step);

    const test_module = b.createModule(.{
        .root_source_file = b.path("tests/all_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tokenizer_module = b.createModule(.{
        .root_source_file = b.path("src/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });

    const transformer_module = b.createModule(.{
        .root_source_file = b.path("src/transformer.zig"),
        .target = target,
        .optimize = optimize,
    });
    const dataset_module = b.createModule(.{
        .root_source_file = b.path("src/dataset.zig"),
        .target = target,
        .optimize = optimize,
    });
    const optimizer_module = b.createModule(.{
        .root_source_file = b.path("src/optimizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    optimizer_module.addImport("transformer", transformer_module);
    const training_module = b.createModule(.{
        .root_source_file = b.path("src/training.zig"),
        .target = target,
        .optimize = optimize,
    });
    training_module.addImport("transformer", transformer_module);

    test_module.addImport("tokenizer", tokenizer_module);
    test_module.addImport("transformer", transformer_module);
    test_module.addImport("dataset", dataset_module);
    test_module.addImport("optimizer", optimizer_module);
    test_module.addImport("training", training_module);

    const tests = b.addTest(.{
        .root_module = test_module,
    });
    tests.test_runner = .{
        .path = b.path("tests/test_runner.zig"),
        .mode = .simple,
    };

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);

    const bench_module = b.createModule(.{
        .root_source_file = b.path("examples/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    const bench_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = bench_module,
    });

    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
