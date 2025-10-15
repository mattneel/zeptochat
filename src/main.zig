const std = @import("std");
const Tokenizer = @import("tokenizer").Tokenizer;
const train_runner = @import("train_runner");
const checkpoint = @import("checkpoint");
const generation = @import("generation");

const TrainConfig = train_runner.TrainConfig;
const CliError = error{ InvalidArgs, InvalidInteger, InvalidFloat };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    const exe = args.next().?;
    const command = args.next() orelse {
        try printUsage(exe);
        return;
    };

    if (std.mem.eql(u8, command, "tokenize")) {
        try tokenizeCommand(allocator, &args);
    } else if (std.mem.eql(u8, command, "train")) {
        try trainCommand(allocator, &args);
    } else if (std.mem.eql(u8, command, "generate")) {
        try generateCommand(allocator, &args);
    } else {
        std.debug.print("Unknown command '{s}'\n\n", .{command});
        try printUsage(exe);
        return;
    }
}

fn printUsage(exe: []const u8) !void {
    std.debug.print("Usage: {s} <command> [args]\n", .{exe});
    std.debug.print("  tokenize <input.txt> <output.tokens> <tokenizer_dir>\n", .{});
    std.debug.print(
        "  train <tokens_file> [epochs] [seq_len] [learning_rate] [save_every] [checkpoint_dir]\n",
        .{},
    );
    std.debug.print("  generate <checkpoint> <tokenizer_dir> <prompt> [max_tokens] [temperature]\n", .{});
}

fn tokenizeCommand(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const input_path = args.next() orelse {
        std.debug.print("Usage: tokenize <input.txt> <output.tokens> <tokenizer_dir>\n", .{});
        return CliError.InvalidArgs;
    };
    const output_path = args.next() orelse {
        std.debug.print("Usage: tokenize <input.txt> <output.tokens> <tokenizer_dir>\n", .{});
        return CliError.InvalidArgs;
    };
    const tokenizer_dir = args.next() orelse {
        std.debug.print("Usage: tokenize <input.txt> <output.tokens> <tokenizer_dir>\n", .{});
        return CliError.InvalidArgs;
    };

    std.log.info("Loading tokenizer from {s}…", .{tokenizer_dir});
    var tokenizer = try loadTokenizer(allocator, tokenizer_dir);
    defer tokenizer.deinit();

    std.log.info("Reading {s}…", .{input_path});
    const text = try std.fs.cwd().readFileAlloc(allocator, input_path, 1024 * 1024 * 1024);
    defer allocator.free(text);

    std.log.info("Tokenizing {} bytes…", .{text.len});
    const tokens = try tokenizer.encode(text);
    defer allocator.free(tokens);

    std.log.info("Writing {} tokens to {s}", .{ tokens.len, output_path });
    const file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(std.mem.sliceAsBytes(tokens));

    std.log.info("done", .{});
}

fn trainCommand(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const tokens_path = args.next() orelse {
        std.debug.print("Usage: train <tokens_file> [epochs] [seq_len] [learning_rate] [save_every] [checkpoint_dir]\n", .{});
        return CliError.InvalidArgs;
    };

    const epochs = try parseOptionalInt(args.next(), 3);
    const seq_len = try parseOptionalInt(args.next(), 64);
    const lr = try parseOptionalFloat(args.next(), 0.001);
    const save_every = try parseOptionalInt(args.next(), 1);
    const checkpoint_dir = args.next() orelse "checkpoints";

    const config = TrainConfig{
        .tokens_path = tokens_path,
        .epochs = epochs,
        .seq_len = seq_len,
        .learning_rate = lr,
        .checkpoint_frequency = save_every,
        .checkpoint_dir = checkpoint_dir,
    };

    try train_runner.run(allocator, config);
}

fn generateCommand(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const checkpoint_path = args.next() orelse {
        std.debug.print("Usage: generate <checkpoint> <tokenizer_dir> <prompt> [max_tokens] [temperature]\n", .{});
        return CliError.InvalidArgs;
    };

    const tokenizer_dir = args.next() orelse {
        std.debug.print("Usage: generate <checkpoint> <tokenizer_dir> <prompt> [max_tokens] [temperature]\n", .{});
        return CliError.InvalidArgs;
    };

    const prompt_text = args.next() orelse {
        std.debug.print("Usage: generate <checkpoint> <tokenizer_dir> <prompt> [max_tokens] [temperature]\n", .{});
        return CliError.InvalidArgs;
    };

    const max_tokens = try parseOptionalInt(args.next(), 128);
    const temperature = try parseOptionalFloat(args.next(), 0.8);

    const loaded = try checkpoint.load(allocator, checkpoint_path);
    var model = loaded.model;
    defer model.deinit();

    std.log.info("Loaded checkpoint at step {d}", .{loaded.meta.step});

    var tokenizer = try loadTokenizer(allocator, tokenizer_dir);
    defer tokenizer.deinit();

    const prompt_tokens = try tokenizer.encode(prompt_text);
    defer allocator.free(prompt_tokens);

    const generated_tokens = try generation.generate(allocator, &model, prompt_tokens, max_tokens, temperature);
    defer allocator.free(generated_tokens);

    const text = try tokenizer.decode(generated_tokens);
    defer allocator.free(text);

    std.debug.print("{s}\n", .{text});
}

fn parseOptionalInt(value: ?[]const u8, default_value: usize) CliError!usize {
    if (value) |text| {
        return std.fmt.parseInt(usize, text, 10) catch return CliError.InvalidInteger;
    }
    return default_value;
}

fn parseOptionalFloat(value: ?[]const u8, default_value: f32) CliError!f32 {
    if (value) |text| {
        return std.fmt.parseFloat(f32, text) catch return CliError.InvalidFloat;
    }
    return default_value;
}

fn loadTokenizer(allocator: std.mem.Allocator, dir_path: []const u8) !Tokenizer {
    var dir = try std.fs.cwd().openDir(dir_path, .{});
    defer dir.close();
    return Tokenizer.initFromDir(allocator, dir, "vocab.txt", "merges.txt");
}
