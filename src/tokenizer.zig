const std = @import("std");

const Error = error{
    InvalidToken,
} || std.mem.Allocator.Error;

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Error!Tokenizer {
        return Tokenizer{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        _ = self;
    }

    pub fn encode(self: *Tokenizer, input: []const u8) Error![]u32 {
        var tokens = try self.allocator.alloc(u32, input.len);
        errdefer self.allocator.free(tokens);

        for (input, 0..) |byte, idx| {
            tokens[idx] = @as(u32, byte);
        }
        return tokens;
    }

    pub fn decode(self: *Tokenizer, tokens: []const u32) Error![]u8 {
        var bytes = try self.allocator.alloc(u8, tokens.len);
        errdefer self.allocator.free(bytes);

        for (tokens, 0..) |token, idx| {
            if (token > std.math.maxInt(u8)) {
                return Error.InvalidToken;
            }
            bytes[idx] = @as(u8, @intCast(token));
        }
        return bytes;
    }
};
