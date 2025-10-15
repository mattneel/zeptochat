const std = @import("std");
const transformer = @import("transformer");

pub const Transformer = transformer.Transformer;
pub const ModelConfig = transformer.ModelConfig;

pub const CheckpointError = error{
    InvalidFile,
    UnexpectedEof,
};

pub const CheckpointMeta = struct {
    config: ModelConfig,
    step: usize,
};

pub fn save(model: *const Transformer, path: []const u8, step: usize) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    try writeInt(file, model.config.vocab_size);
    try writeInt(file, model.config.context_length);
    try writeInt(file, model.config.d_model);
    try writeInt(file, model.config.n_heads);
    try writeInt(file, model.config.n_layers);
    try writeFloat(file, model.config.dropout);
    try writeInt(file, step);

    try file.writeAll(std.mem.sliceAsBytes(model.token_embeddings));
    try file.writeAll(std.mem.sliceAsBytes(model.position_embeddings));

    for (model.layers) |layer| {
        try file.writeAll(std.mem.sliceAsBytes(layer.ln1.weight));
        try file.writeAll(std.mem.sliceAsBytes(layer.ln1.bias));
        try file.writeAll(std.mem.sliceAsBytes(layer.attn.w_q));
        try file.writeAll(std.mem.sliceAsBytes(layer.attn.w_k));
        try file.writeAll(std.mem.sliceAsBytes(layer.attn.w_v));
        try file.writeAll(std.mem.sliceAsBytes(layer.attn.w_o));
        try file.writeAll(std.mem.sliceAsBytes(layer.ln2.weight));
        try file.writeAll(std.mem.sliceAsBytes(layer.ln2.bias));
        try file.writeAll(std.mem.sliceAsBytes(layer.mlp.w1));
        try file.writeAll(std.mem.sliceAsBytes(layer.mlp.b1));
        try file.writeAll(std.mem.sliceAsBytes(layer.mlp.w2));
        try file.writeAll(std.mem.sliceAsBytes(layer.mlp.b2));
    }

    try file.writeAll(std.mem.sliceAsBytes(model.ln_final.weight));
    try file.writeAll(std.mem.sliceAsBytes(model.ln_final.bias));
    try file.writeAll(std.mem.sliceAsBytes(model.lm_head));
}

pub fn load(allocator: std.mem.Allocator, path: []const u8) !struct {
    meta: CheckpointMeta,
    model: Transformer,
} {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const size = try file.getEndPos();
    try file.seekTo(0);

    const data = try allocator.alloc(u8, size);
    defer allocator.free(data);
    if ((try file.readAll(data)) != size) return CheckpointError.UnexpectedEof;

    var cursor: usize = 0;

    const vocab_size = try readIntSlice(data, &cursor);
    const context_length = try readIntSlice(data, &cursor);
    const d_model = try readIntSlice(data, &cursor);
    const n_heads = try readIntSlice(data, &cursor);
    const n_layers = try readIntSlice(data, &cursor);

    const dropout = try readFloatSlice(data, &cursor);

    const step = try readIntSlice(data, &cursor);

    const config = ModelConfig{
        .vocab_size = vocab_size,
        .context_length = context_length,
        .d_model = d_model,
        .n_heads = n_heads,
        .n_layers = n_layers,
        .dropout = dropout,
    };

    var model = try Transformer.init(allocator, config);
    errdefer model.deinit();

    try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(model.token_embeddings));
    try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(model.position_embeddings));

    for (model.layers) |*layer| {
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.ln1.weight));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.ln1.bias));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.attn.w_q));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.attn.w_k));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.attn.w_v));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.attn.w_o));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.ln2.weight));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.ln2.bias));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.mlp.w1));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.mlp.b1));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.mlp.w2));
        try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(layer.mlp.b2));
    }

    try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(model.ln_final.weight));
    try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(model.ln_final.bias));
    try readIntoSlice(data, &cursor, std.mem.sliceAsBytes(model.lm_head));

    return .{
        .meta = .{ .config = config, .step = step },
        .model = model,
    };
}

fn writeInt(file: std.fs.File, value: usize) !void {
    var buf: [@sizeOf(usize)]u8 = undefined;
    std.mem.writeInt(usize, &buf, value, .little);
    try file.writeAll(&buf);
}

fn writeFloat(file: std.fs.File, value: f32) !void {
    var buf: [@sizeOf(f32)]u8 = undefined;
    const bits: u32 = @bitCast(value);
    std.mem.writeInt(u32, &buf, bits, .little);
    try file.writeAll(&buf);
}

fn readIntoSlice(data: []const u8, cursor: *usize, dest: []u8) !void {
    const end = cursor.* + dest.len;
    if (end > data.len) return CheckpointError.UnexpectedEof;
    std.mem.copyForwards(u8, dest, data[cursor.*..end]);
    cursor.* = end;
}

fn readIntSlice(data: []const u8, cursor: *usize) !usize {
    var buf: [@sizeOf(usize)]u8 = undefined;
    try readIntoSlice(data, cursor, &buf);
    return std.mem.readInt(usize, &buf, .little);
}

fn readFloatSlice(data: []const u8, cursor: *usize) !f32 {
    var buf: [@sizeOf(f32)]u8 = undefined;
    try readIntoSlice(data, cursor, &buf);
    const bits = std.mem.readInt(u32, &buf, .little);
    return @bitCast(bits);
}
