const std = @import("std");
const transformer = @import("transformer");

const Transformer = transformer.Transformer;

pub fn generate(
    allocator: std.mem.Allocator,
    model: *Transformer,
    prompt: []const u32,
    max_tokens: usize,
    temperature: f32,
) ![]u32 {
    var list = std.ArrayListUnmanaged(u32){};
    errdefer list.deinit(allocator);
    try list.ensureTotalCapacity(allocator, prompt.len + max_tokens);
    try list.appendSlice(allocator, prompt);

    const timestamp = std.time.milliTimestamp();
    const seed = @as(u64, @intCast(timestamp));
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();

    const context_limit = model.config.context_length;

    for (0..max_tokens) |_| {
        const available = list.items.len;
        const context_start = if (available > context_limit) available - context_limit else 0;
        const context = list.items[context_start..];

        const logits = try model.forward(context);
        defer allocator.free(logits);

        const vocab_size = model.config.vocab_size;
        const offset = (context.len - 1) * vocab_size;
        const last_logits = logits[offset .. offset + vocab_size];

        const next_token = try sampleToken(allocator, last_logits, temperature, random);
        try list.append(allocator, next_token);
    }

    return list.toOwnedSlice(allocator);
}

fn sampleToken(
    allocator: std.mem.Allocator,
    logits: []const f32,
    temperature: f32,
    random: std.Random,
) !u32 {
    if (temperature <= 0) {
        return argMax(logits);
    }

    var probs = try allocator.alloc(f32, logits.len);
    defer allocator.free(probs);

    var max_logit: f32 = -std.math.inf(f32);
    for (logits) |value| max_logit = @max(max_logit, value);

    var sum: f32 = 0;
    for (logits, 0..) |value, i| {
        const adjusted = (value - max_logit) / temperature;
        const p = @exp(adjusted);
        probs[i] = p;
        sum += p;
    }

    if (sum == 0) return argMax(logits);

    for (probs) |*p| p.* /= sum;

    const r = random.float(f32);
    var cumulative: f32 = 0;
    for (probs, 0..) |p, i| {
        cumulative += p;
        if (r <= cumulative) return @as(u32, @intCast(i));
    }

    return @as(u32, @intCast(probs.len - 1));
}

fn argMax(values: []const f32) u32 {
    var index: usize = 0;
    var best = values[0];
    for (values[1..], 1..) |value, i| {
        if (value > best) {
            best = value;
            index = i;
        }
    }
    return @as(u32, @intCast(index));
}
