const std = @import("std");

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    work_queue: WorkQueue,
    shutdown: std.atomic.Value(bool),
    pending: std.atomic.Value(usize),
    started: bool,

    const WorkQueue = struct {
        allocator: std.mem.Allocator,
        mutex: std.Thread.Mutex = .{},
        condition: std.Thread.Condition = .{},
        jobs: std.ArrayListUnmanaged(Job),

        const Job = struct {
            func: *const fn (*anyopaque) void,
            data: *anyopaque,
        };

        fn init(allocator: std.mem.Allocator) WorkQueue {
            return .{
                .allocator = allocator,
                .jobs = .{},
            };
        }

        fn deinit(self: *WorkQueue) void {
            self.jobs.deinit(self.allocator);
        }

        fn push(self: *WorkQueue, job: Job) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            try self.jobs.append(self.allocator, job);
            self.condition.signal();
        }

        fn popLocked(self: *WorkQueue) Job {
            std.debug.assert(self.jobs.items.len > 0);
            return self.jobs.orderedRemove(0);
        }

        fn isEmpty(self: *WorkQueue) bool {
            return self.jobs.items.len == 0;
        }
    };

    pub fn init(allocator: std.mem.Allocator, num_threads: usize) !ThreadPool {
        const desired = if (num_threads == 0)
            try std.Thread.getCpuCount()
        else
            num_threads;

        const threads = try allocator.alloc(std.Thread, desired);
        errdefer allocator.free(threads);

        var pool = ThreadPool{
            .allocator = allocator,
            .threads = threads,
            .work_queue = WorkQueue.init(allocator),
            .shutdown = std.atomic.Value(bool).init(false),
            .pending = std.atomic.Value(usize).init(0),
            .started = false,
        };

        errdefer pool.work_queue.deinit();

        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        if (self.started) {
            self.shutdown.store(true, .release);
            self.work_queue.condition.broadcast();

            for (self.threads) |thread| {
                thread.join();
            }
        }

        self.allocator.free(self.threads);
        self.work_queue.deinit();
    }

    fn start(self: *ThreadPool) !void {
        if (self.started) return;

        self.shutdown.store(false, .release);

        var spawned: usize = 0;
        errdefer {
            self.shutdown.store(true, .release);
            self.work_queue.condition.broadcast();
            for (self.threads[0..spawned]) |thread| thread.join();
        }

        while (spawned < self.threads.len) : (spawned += 1) {
            self.threads[spawned] = try std.Thread.spawn(.{}, workerLoop, .{self});
        }

        self.started = true;
    }

    pub fn threadCount(self: *ThreadPool) usize {
        return self.threads.len;
    }

    pub fn submit(
        self: *ThreadPool,
        comptime T: type,
        func: fn (*T) void,
        data: *T,
    ) !void {
        const Wrapper = struct {
            fn call(ctx: *anyopaque) void {
                const typed: *T = @ptrCast(@alignCast(ctx));
                func(typed);
            }
        };

        const job = WorkQueue.Job{
            .func = Wrapper.call,
            .data = @ptrCast(@alignCast(data)),
        };

        if (!self.started) try self.start();
        _ = self.pending.fetchAdd(1, .acq_rel);
        self.work_queue.push(job) catch |err| {
            _ = self.pending.fetchSub(1, .acq_rel);
            return err;
        };
    }

    pub fn wait(self: *ThreadPool) void {
        while (true) {
            if (self.pending.load(.acquire) == 0) {
                self.work_queue.mutex.lock();
                const empty = self.work_queue.isEmpty();
                self.work_queue.mutex.unlock();
                if (empty) break;
            }
            std.Thread.sleep(500_000);
        }
    }

    pub fn parallelFor(
        self: *ThreadPool,
        comptime func: fn (usize) void,
        begin_index: usize,
        end: usize,
    ) !void {
        if (begin_index >= end) return;

        const length = end - begin_index;
        const workers = @max(@as(usize, 1), @min(self.threadCount(), length));
        const chunk = @max(@as(usize, 1), length / workers);

        const RangeCtx = struct {
            pool: *ThreadPool,
            start: usize,
            end: usize,
        };

        const Runner = struct {
            fn run(ctx_ptr: *RangeCtx) void {
                const ctx = ctx_ptr.*;
                defer ctx.pool.allocator.destroy(ctx_ptr);

                var idx = ctx.start;
                while (idx < ctx.end) : (idx += 1) {
                    func(idx);
                }
            }
        };

        var i = begin_index;
        while (i < end) {
            const chunk_end = @min(i + chunk, end);
            const ctx = try self.allocator.create(RangeCtx);
            ctx.* = .{ .pool = self, .start = i, .end = chunk_end };
            try self.submit(RangeCtx, Runner.run, ctx);
            i = chunk_end;
        }

        self.wait();
    }

    pub fn parallelMap(
        self: *ThreadPool,
        comptime T: type,
        comptime U: type,
        comptime func: fn (T) U,
        input: []const T,
        output: []U,
    ) !void {
        std.debug.assert(input.len == output.len);
        if (input.len == 0) return;

        const workers = @max(@as(usize, 1), @min(self.threadCount(), input.len));
        const chunk = @max(@as(usize, 1), input.len / workers);

        const ChunkCtx = struct {
            pool: *ThreadPool,
            input_slice: []const T,
            output_slice: []U,
        };

        const Runner = struct {
            fn run(ctx_ptr: *ChunkCtx) void {
                const ctx = ctx_ptr.*;
                defer ctx.pool.allocator.destroy(ctx_ptr);

                for (ctx.input_slice, ctx.output_slice) |value, *out_ptr| {
                    out_ptr.* = func(value);
                }
            }
        };

        var index: usize = 0;
        while (index < input.len) {
            const end = @min(index + chunk, input.len);
            const ctx = try self.allocator.create(ChunkCtx);
            ctx.* = .{
                .pool = self,
                .input_slice = input[index..end],
                .output_slice = output[index..end],
            };
            try self.submit(ChunkCtx, Runner.run, ctx);
            index = end;
        }

        self.wait();
    }

    fn workerLoop(pool: *ThreadPool) void {
        while (true) {
            pool.work_queue.mutex.lock();
            while (pool.work_queue.isEmpty()) {
                if (pool.shutdown.load(.acquire)) {
                    pool.work_queue.mutex.unlock();
                    return;
                }
                pool.work_queue.condition.wait(&pool.work_queue.mutex);
            }

            const job = pool.work_queue.popLocked();
            pool.work_queue.mutex.unlock();

            job.func(job.data);
            _ = pool.pending.fetchSub(1, .acq_rel);
        }
    }
};

pub fn parallelRange(
    allocator: std.mem.Allocator,
    comptime func: fn (usize) void,
    start: usize,
    end: usize,
    num_threads: usize,
) !void {
    if (start >= end) return;

    const total = end - start;
    const actual_threads = @max(@as(usize, 1), @min(num_threads, total));
    const chunk = @max(@as(usize, 1), total / actual_threads);

    const Range = struct {
        start: usize,
        end: usize,
    };

    const threads = try allocator.alloc(std.Thread, actual_threads);
    defer allocator.free(threads);

    const ranges = try allocator.alloc(Range, actual_threads);
    defer allocator.free(ranges);

    const Runner = struct {
        fn call(range: Range) void {
            for (range.start..range.end) |idx| {
                func(idx);
            }
        }
    };

    for (threads, 0..) |*thread, i| {
        const range_start = start + i * chunk;
        const range_end = if (i == actual_threads - 1) end else @min(range_start + chunk, end);
        ranges[i] = .{ .start = range_start, .end = range_end };
        thread.* = try std.Thread.spawn(.{}, Runner.call, .{ranges[i]});
    }

    for (threads) |thread| {
        thread.join();
    }
}
