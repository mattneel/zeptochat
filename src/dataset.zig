const std = @import("std");

pub const DatasetError = error{
    InvalidTokenFile,
    IncompleteRead,
};

pub const Dataset = struct {
    tokens: []u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !Dataset {
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const info = try file.stat();
        if (info.size % @sizeOf(u32) != 0) {
            return DatasetError.InvalidTokenFile;
        }

        const count = info.size / @sizeOf(u32);
        const tokens = try allocator.alloc(u32, count);
        errdefer allocator.free(tokens);

        const bytes_read = try file.readAll(std.mem.sliceAsBytes(tokens));
        if (bytes_read != info.size) {
            return DatasetError.IncompleteRead;
        }

        return Dataset{
            .tokens = tokens,
            .allocator = allocator,
        };
    }

    pub fn initFromSlice(allocator: std.mem.Allocator, source: []const u32) !Dataset {
        const tokens = try allocator.alloc(u32, source.len);
        @memcpy(tokens, source);
        return Dataset{
            .tokens = tokens,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Dataset) void {
        self.allocator.free(self.tokens);
        self.* = undefined;
    }

    pub fn iterator(self: *Dataset, batch_size: usize, seq_len: usize) Iterator {
        return Iterator{
            .dataset = self,
            .batch_size = batch_size,
            .seq_len = seq_len,
            .position = 0,
        };
    }
};

pub const Batch = struct {
    tokens: []const u32,
    targets: []const u32,
};

pub const Iterator = struct {
    dataset: *Dataset,
    batch_size: usize,
    seq_len: usize,
    position: usize,

    pub fn next(self: *Iterator) ?Batch {
        if (self.seq_len == 0 or self.batch_size == 0) {
            return null;
        }

        const tokens_per_batch = self.batch_size * self.seq_len;
        const required = self.position + tokens_per_batch + 1;
        if (required > self.dataset.tokens.len) {
            return null;
        }

        const tokens_slice = self.dataset.tokens[self.position .. self.position + tokens_per_batch];
        const targets_slice = self.dataset.tokens[self.position + 1 .. self.position + tokens_per_batch + 1];
        self.position += tokens_per_batch;

        return Batch{
            .tokens = tokens_slice,
            .targets = targets_slice,
        };
    }

    pub fn reset(self: *Iterator) void {
        self.position = 0;
    }

    pub fn remaining(self: *Iterator) usize {
        const tokens_per_batch = self.batch_size * self.seq_len;
        if (tokens_per_batch == 0 or self.dataset.tokens.len <= self.position) {
            return 0;
        }
        const available = self.dataset.tokens.len - self.position - 1;
        return available / tokens_per_batch;
    }
};
