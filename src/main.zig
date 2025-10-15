const std = @import("std");
const clap = @import("clap");
const tokenizer_mod = @import("tokenizer");
const train_runner = @import("train_runner");
const checkpoint = @import("checkpoint");
const generation = @import("generation");
const transformer = @import("transformer");

const Tokenizer = tokenizer_mod.Tokenizer;

const VERSION = "0.1.0";
const DEFAULT_TOKENIZER_DIR = "tests/fixtures/gpt2_mini";

const SubCommand = enum {
    tokenize,
    train,
    generate,
};

const main_params = clap.parseParamsComptime(
    \\-h, --help     Display this help and exit.
    \\-V, --version  Display version information and exit.
    \\<command>
    \\
);

const main_parsers = .{
    .command = clap.parsers.enumeration(SubCommand),
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var iter = try std.process.ArgIterator.initWithAllocator(allocator);
    defer iter.deinit();

    _ = iter.next(); // consume executable name

    var diag = clap.Diagnostic{};
    var res = clap.parseEx(clap.Help, &main_params, main_parsers, &iter, .{
        .diagnostic = &diag,
        .allocator = allocator,
        .terminating_positional = 0,
    }) catch |err| {
        try diag.reportToFile(.stderr(), err);
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        try clap.helpToFile(.stdout(), clap.Help, &main_params, .{});
        return;
    }
    if (res.args.version != 0) {
        std.debug.print("zeptochat {s}\n", .{VERSION});
        return;
    }

    const command = res.positionals[0] orelse {
        try clap.usageToFile(.stderr(), clap.Help, &main_params);
        return;
    };

    switch (command) {
        .tokenize => try tokenizeMain(allocator, &iter),
        .train => try trainMain(allocator, &iter),
        .generate => try generateMain(allocator, &iter),
    }
}

fn tokenizeMain(allocator: std.mem.Allocator, iter: *std.process.ArgIterator) !void {
    const params = comptime clap.parseParamsComptime(
        \\-h, --help                    Display this help and exit.
        \\-t, --tokenizer-dir <PATH>    Tokenizer directory (default: tests/fixtures/gpt2_mini).
        \\<INPUT>
        \\<OUTPUT>
        \\
    );
    const parsers = .{
        .PATH = clap.parsers.string,
        .INPUT = clap.parsers.string,
        .OUTPUT = clap.parsers.string,
        .usize = clap.parsers.int(usize, 10),
        .f32 = clap.parsers.float(f32),
    };

    var diag = clap.Diagnostic{};
    var res = clap.parseEx(clap.Help, &params, parsers, iter, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        try diag.reportToFile(.stderr(), err);
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        try clap.helpToFile(.stdout(), clap.Help, &params, .{});
        return;
    }

    const input_path = res.positionals[0] orelse {
        try clap.usageToFile(.stderr(), clap.Help, &params);
        return;
    };
    const output_path = res.positionals[1] orelse {
        try clap.usageToFile(.stderr(), clap.Help, &params);
        return;
    };
    const tokenizer_dir = res.args.@"tokenizer-dir" orelse DEFAULT_TOKENIZER_DIR;
    try tokenizeFile(allocator, tokenizer_dir, input_path, output_path);
}

fn tokenizeFile(
    allocator: std.mem.Allocator,
    tokenizer_dir: []const u8,
    input_path: []const u8,
    output_path: []const u8,
) !void {
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

fn trainMain(allocator: std.mem.Allocator, iter: *std.process.ArgIterator) !void {
    const params = comptime clap.parseParamsComptime(
        \\-h, --help                      Display this help and exit.
        \\-c, --config <PATH>             Load training configuration from JSON.
        \\    --epochs <usize>            Number of epochs.
        \\    --seq-len <usize>           Sequence length override.
        \\    --learning-rate <f32>       Optimizer learning rate.
        \\    --batch-size <usize>        Batch size.
        \\    --save-every <usize>        Checkpoint frequency in epochs.
        \\    --checkpoint-dir <PATH>     Output directory for checkpoints.
        \\    --checkpoint-prefix <PATH>  Prefix for checkpoint filenames.
        \\<TOKENS>
        \\
    );
    const parsers = .{
        .PATH = clap.parsers.string,
        .TOKENS = clap.parsers.string,
        .usize = clap.parsers.int(usize, 10),
        .f32 = clap.parsers.float(f32),
    };

    var diag = clap.Diagnostic{};
    var res = clap.parseEx(clap.Help, &params, parsers, iter, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        try diag.reportToFile(.stderr(), err);
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        try clap.helpToFile(.stdout(), clap.Help, &params, .{});
        return;
    }

    const tokens_path = res.positionals[0] orelse {
        try clap.usageToFile(.stderr(), clap.Help, &params);
        return;
    };

    var config = train_runner.TrainConfig{
        .tokens_path = tokens_path,
    };

    var owned_strings = std.ArrayListUnmanaged([]u8){};
    defer {
        for (owned_strings.items) |buf| allocator.free(buf);
        owned_strings.deinit(allocator);
    }

    var seq_len_overridden = false;
    if (res.args.config) |config_path| {
        try loadTrainConfigFromJson(allocator, config_path, &config, &owned_strings, &seq_len_overridden);
    }

    if (res.args.epochs) |value| config.epochs = value;
    if (res.args.@"seq-len") |value| {
        config.seq_len = value;
        seq_len_overridden = true;
    }
    if (res.args.@"learning-rate") |value| config.learning_rate = value;
    if (res.args.@"batch-size") |value| config.batch_size = value;
    if (res.args.@"save-every") |value| config.checkpoint_frequency = value;
    if (res.args.@"checkpoint-dir") |dir| config.checkpoint_dir = dir;
    if (res.args.@"checkpoint-prefix") |prefix| config.checkpoint_prefix = prefix;

    if (!seq_len_overridden and config.model.context_length != transformer.TinyConfig.context_length) {
        config.seq_len = config.model.context_length;
    }

    try train_runner.run(allocator, config);
}

fn generateMain(allocator: std.mem.Allocator, iter: *std.process.ArgIterator) !void {
    const params = comptime clap.parseParamsComptime(
        \\-h, --help                    Display this help and exit.
        \\-t, --tokenizer-dir <PATH>    Tokenizer directory (default: tests/fixtures/gpt2_mini).
        \\    --max-tokens <usize>      Number of tokens to generate (default: 128).
        \\    --temperature <f32>       Sampling temperature (default: 0.8).
        \\<CHECKPOINT>
        \\<PROMPT>
        \\
    );
    const parsers = .{
        .PATH = clap.parsers.string,
        .CHECKPOINT = clap.parsers.string,
        .PROMPT = clap.parsers.string,
        .usize = clap.parsers.int(usize, 10),
        .f32 = clap.parsers.float(f32),
    };

    var diag = clap.Diagnostic{};
    var res = clap.parseEx(clap.Help, &params, parsers, iter, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        try diag.reportToFile(.stderr(), err);
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        try clap.helpToFile(.stdout(), clap.Help, &params, .{});
        return;
    }

    const checkpoint_path = res.positionals[0] orelse {
        try clap.usageToFile(.stderr(), clap.Help, &params);
        return;
    };
    const prompt_text = res.positionals[1] orelse {
        try clap.usageToFile(.stderr(), clap.Help, &params);
        return;
    };

    const tokenizer_dir = res.args.@"tokenizer-dir" orelse DEFAULT_TOKENIZER_DIR;
    const max_tokens = res.args.@"max-tokens" orelse 128;
    const temperature = res.args.temperature orelse 0.8;

    const loaded = try checkpoint.load(allocator, checkpoint_path);
    var model = loaded.model;
    defer model.deinit();

    std.log.info("Loaded checkpoint at step {d}", .{loaded.meta.step});

    var tokenizer = try loadTokenizer(allocator, tokenizer_dir);
    defer tokenizer.deinit();

    const prompt_tokens = try tokenizer.encode(prompt_text);
    defer allocator.free(prompt_tokens);

    const generated_tokens = try generation.generate(
        allocator,
        &model,
        prompt_tokens,
        max_tokens,
        temperature,
    );
    defer allocator.free(generated_tokens);

    const text = try tokenizer.decode(generated_tokens);
    defer allocator.free(text);

    std.debug.print("{s}\n", .{text});
}

fn loadTrainConfigFromJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: *train_runner.TrainConfig,
    owned_strings: *std.ArrayListUnmanaged([]u8),
    seq_len_overridden: *bool,
) !void {
    std.log.info("Loading training config from {s}", .{path});
    const json_bytes = try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
    defer allocator.free(json_bytes);

    var parsed = try std.json.parseFromSlice(TrainConfigFile, allocator, json_bytes, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const value = parsed.value;

    if (value.tokens_path) |tokens| try assignString(allocator, owned_strings, &config.tokens_path, tokens);
    if (value.epochs) |epochs| config.epochs = epochs;
    if (value.seq_len) |seq_len| {
        config.seq_len = seq_len;
        seq_len_overridden.* = true;
    }
    if (value.learning_rate) |lr| config.learning_rate = lr;
    if (value.batch_size) |bs| config.batch_size = bs;
    if (value.checkpoint_dir) |dir| try assignString(allocator, owned_strings, &config.checkpoint_dir, dir);
    if (value.checkpoint_prefix) |prefix| try assignString(allocator, owned_strings, &config.checkpoint_prefix, prefix);
    if (value.checkpoint_frequency) |freq| config.checkpoint_frequency = freq;
    if (value.model) |model_json| {
        var model_config = config.model;
        applyModelJson(&model_config, model_json);
        config.model = model_config;
    }
}

fn assignString(
    allocator: std.mem.Allocator,
    owned_strings: *std.ArrayList([]u8),
    dest: *[]const u8,
    value: []const u8,
) !void {
    const dup = try allocator.dupe(u8, value);
    try owned_strings.append(allocator, dup);
    dest.* = dup;
}

fn applyModelJson(model: *transformer.ModelConfig, value: ModelConfigFile) void {
    if (value.vocab_size) |v| model.vocab_size = v;
    if (value.context_length) |v| model.context_length = v;
    if (value.d_model) |v| model.d_model = v;
    if (value.n_heads) |v| model.n_heads = v;
    if (value.n_layers) |v| model.n_layers = v;
    if (value.dropout) |v| model.dropout = v;
}

fn loadTokenizer(allocator: std.mem.Allocator, dir_path: []const u8) !Tokenizer {
    var dir = try std.fs.cwd().openDir(dir_path, .{});
    defer dir.close();
    return Tokenizer.initFromDir(allocator, dir, "vocab.txt", "merges.txt");
}

const TrainConfigFile = struct {
    tokens_path: ?[]const u8 = null,
    epochs: ?usize = null,
    seq_len: ?usize = null,
    learning_rate: ?f32 = null,
    batch_size: ?usize = null,
    checkpoint_dir: ?[]const u8 = null,
    checkpoint_prefix: ?[]const u8 = null,
    checkpoint_frequency: ?usize = null,
    model: ?ModelConfigFile = null,
};

const ModelConfigFile = struct {
    vocab_size: ?usize = null,
    context_length: ?usize = null,
    d_model: ?usize = null,
    n_heads: ?usize = null,
    n_layers: ?usize = null,
    dropout: ?f32 = null,
};
