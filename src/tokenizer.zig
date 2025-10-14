const std = @import("std");

const Error = error{
    InvalidToken,
    UnknownToken,
    DuplicateToken,
    DuplicateMerge,
    InvalidFormat,
} || std.mem.Allocator.Error || std.fs.File.OpenError || std.fs.File.ReadError || std.fs.File.SeekError;

const Merge = struct {
    rank: u32,
    result_id: u32,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    token_lookup: std.StringHashMap(u32),
    token_bytes: std.AutoHashMap(u32, []u8),
    merges: std.AutoHashMap(u64, Merge),
    next_merge_rank: u32,

    pub fn init(allocator: std.mem.Allocator) Error!Tokenizer {
        return Tokenizer{
            .allocator = allocator,
            .token_lookup = std.StringHashMap(u32).init(allocator),
            .token_bytes = std.AutoHashMap(u32, []u8).init(allocator),
            .merges = std.AutoHashMap(u64, Merge).init(allocator),
            .next_merge_rank = 0,
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

        var all_digits = true;
        for (symbol) |ch| {
            if (!std.ascii.isDigit(ch)) {
                all_digits = false;
                break;
            }
        }

        if (all_digits) {
            const id = std.fmt.parseInt(u32, symbol, 10) catch return Error.UnknownToken;
            if (id <= std.math.maxInt(u8)) {
                return id;
            }
            if (self.token_bytes.contains(id)) {
                return id;
            }
            return Error.UnknownToken;
        }

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
        if (params.rank >= self.next_merge_rank) {
            self.next_merge_rank = params.rank + 1;
        }
    }

    pub fn loadVocabFromBytes(self: *Tokenizer, bytes: []const u8) Error!void {
        var line_it = std.mem.splitScalar(u8, bytes, '\n');
        while (line_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;
            if (trimmed[0] == '#') continue;

            const sep = std.mem.indexOfAny(u8, trimmed, " \t") orelse return Error.InvalidFormat;
            const id_slice = trimmed[0..sep];
            const symbol_slice = std.mem.trim(u8, trimmed[sep + 1 ..], " \t");
            if (symbol_slice.len == 0) return Error.InvalidFormat;

            const id = std.fmt.parseInt(u32, id_slice, 10) catch return Error.InvalidFormat;
            try self.addToken(symbol_slice, id);
        }
    }

    pub fn loadMergesFromBytes(self: *Tokenizer, bytes: []const u8) Error!void {
        var auto_rank = self.next_merge_rank;
        var line_it = std.mem.splitScalar(u8, bytes, '\n');
        while (line_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;
            if (trimmed[0] == '#') continue;

            var parts: [4][]const u8 = undefined;
            var count: usize = 0;

            var token_it = std.mem.tokenizeAny(u8, trimmed, " \t");
            while (token_it.next()) |part| {
                if (count == parts.len) break;
                parts[count] = part;
                count += 1;
            }

            if (count < 3) {
                return Error.InvalidFormat;
            }

            var rank: u32 = undefined;
            if (count >= 4) {
                rank = std.fmt.parseInt(u32, parts[3], 10) catch return Error.InvalidFormat;
            } else {
                rank = auto_rank;
                auto_rank += 1;
            }

            try self.addMerge(.{
                .left = parts[0],
                .right = parts[1],
                .result = parts[2],
                .rank = rank,
            });
        }

        if (auto_rank > self.next_merge_rank) {
            self.next_merge_rank = auto_rank;
        }
    }

    fn readFileAlloc(self: *Tokenizer, dir: std.fs.Dir, path: []const u8, max_bytes: usize) Error![]u8 {
        return dir.readFileAlloc(self.allocator, path, max_bytes) catch |err| {
            if (err == error.FileTooBig) {
                return Error.InvalidFormat;
            }
            return err;
        };
    }

    pub fn loadVocabFromFile(self: *Tokenizer, dir: std.fs.Dir, path: []const u8) Error!void {
        const bytes = try self.readFileAlloc(dir, path, 10 * 1024 * 1024);
        defer self.allocator.free(bytes);
        try self.loadVocabFromBytes(bytes);
    }

    pub fn loadMergesFromFile(self: *Tokenizer, dir: std.fs.Dir, path: []const u8) Error!void {
        const bytes = try self.readFileAlloc(dir, path, 10 * 1024 * 1024);
        defer self.allocator.free(bytes);
        try self.loadMergesFromBytes(bytes);
    }

    pub fn initFromFiles(
        allocator: std.mem.Allocator,
        vocab_path: []const u8,
        merges_path: []const u8,
        dir: std.fs.Dir,
    ) Error!Tokenizer {
        var tokenizer = try Tokenizer.init(allocator);
        errdefer tokenizer.deinit();

        try tokenizer.loadVocabFromFile(dir, vocab_path);
        try tokenizer.loadMergesFromFile(dir, merges_path);
        return tokenizer;
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
