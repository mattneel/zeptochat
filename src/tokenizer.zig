const std = @import("std");

const Error = error{
    InvalidToken,
    UnknownToken,
    DuplicateToken,
    DuplicateMerge,
} || std.mem.Allocator.Error;

const Merge = struct {
    rank: u32,
    result_id: u32,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    token_lookup: std.StringHashMap(u32),
    token_bytes: std.AutoHashMap(u32, []u8),
    merges: std.AutoHashMap(u64, Merge),

    pub fn init(allocator: std.mem.Allocator) Error!Tokenizer {
        return Tokenizer{
            .allocator = allocator,
            .token_lookup = std.StringHashMap(u32).init(allocator),
            .token_bytes = std.AutoHashMap(u32, []u8).init(allocator),
            .merges = std.AutoHashMap(u64, Merge).init(allocator),
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        var it = self.token_bytes.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.token_bytes.deinit();
        self.token_lookup.deinit();
        self.merges.deinit();
    }

    fn pairKey(left: u32, right: u32) u64 {
        return (@as(u64, left) << 32) | @as(u64, right);
    }

    fn resolveTokenId(self: *Tokenizer, symbol: []const u8) Error!u32 {
        if (symbol.len == 0) return Error.UnknownToken;
        if (symbol.len == 1) {
            return @as(u32, symbol[0]);
        }
        if (self.token_lookup.get(symbol)) |id| {
            return id;
        }
        return Error.UnknownToken;
    }

    pub fn addToken(self: *Tokenizer, symbol: []const u8, id: u32) Error!void {
        if (symbol.len == 0) {
            return Error.UnknownToken;
        }
        if (self.token_lookup.get(symbol)) |_| {
            return Error.DuplicateToken;
        }
        if (self.token_bytes.contains(id)) {
            return Error.DuplicateToken;
        }

        const dup = try self.allocator.dupe(u8, symbol);
        errdefer self.allocator.free(dup);

        try self.token_lookup.put(dup, id);
        errdefer _ = self.token_lookup.remove(dup);

        self.token_bytes.put(id, dup) catch |err| {
            _ = self.token_lookup.remove(dup);
            self.allocator.free(dup);
            return err;
        };
    }

    pub fn addMerge(
        self: *Tokenizer,
        params: struct {
            left: []const u8,
            right: []const u8,
            result: []const u8,
            rank: u32,
        },
    ) Error!void {
        const left_id = try self.resolveTokenId(params.left);
        const right_id = try self.resolveTokenId(params.right);
        const result_id = try self.resolveTokenId(params.result);

        const key = pairKey(left_id, right_id);
        if (self.merges.contains(key)) {
            return Error.DuplicateMerge;
        }

        try self.merges.put(key, .{
            .rank = params.rank,
            .result_id = result_id,
        });
    }

    pub fn encode(self: *Tokenizer, input: []const u8) Error![]u32 {
        var working = std.ArrayList(u32){};
        defer working.deinit(self.allocator);

        for (input) |byte| {
            try working.append(self.allocator, @as(u32, byte));
        }

        if (working.items.len == 0) {
            return self.allocator.alloc(u32, 0);
        }

        while (working.items.len >= 2) {
            var best_idx: usize = 0;
            var best_rank: u32 = std.math.maxInt(u32);
            var found = false;

            const stop = working.items.len - 1;
            var i: usize = 0;
            while (i < stop) : (i += 1) {
                const left = working.items[i];
                const right = working.items[i + 1];
                if (self.merges.get(pairKey(left, right))) |merge| {
                    if (!found or merge.rank < best_rank) {
                        found = true;
                        best_rank = merge.rank;
                        best_idx = i;
                    }
                }
            }

            if (!found) break;

            const merge = self.merges.get(pairKey(working.items[best_idx], working.items[best_idx + 1])).?;
            working.items[best_idx] = merge.result_id;
            _ = working.orderedRemove(best_idx + 1);
        }

        return working.toOwnedSlice(self.allocator);
    }

    pub fn decode(self: *Tokenizer, tokens: []const u32) Error![]u8 {
        var output = std.ArrayList(u8){};
        defer output.deinit(self.allocator);

        for (tokens) |token| {
            if (token <= std.math.maxInt(u8)) {
                try output.append(self.allocator, @as(u8, @intCast(token)));
                continue;
            }

            if (self.token_bytes.get(token)) |bytes_ref| {
                try output.appendSlice(self.allocator, bytes_ref);
                continue;
            }

            return Error.InvalidToken;
        }

        return output.toOwnedSlice(self.allocator);
    }
};
