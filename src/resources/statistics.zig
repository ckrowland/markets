const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Self = @This();

num_transactions: array(u32),
second: f32 = 0,
num_empty_consumers: array(u32),
num_total_producer_inventory: array(u32),

const StagingBuffer = struct {
    slice: ?[]const u32 = null,
    buffer: wgpu.Buffer = undefined,
};

pub const NUM_STATS = 8;
pub const zero = [NUM_STATS]u32{ 0, 0, 0, 0, 0, 0, 0, 0 };

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .num_transactions = array(u32).init(allocator),
        .num_empty_consumers = array(u32).init(allocator),
        .num_total_producer_inventory = array(u32).init(allocator),
    };
}

pub fn deinit(self: *Self) void {
    self.num_transactions.deinit();
    self.num_empty_consumers.deinit();
    self.num_total_producer_inventory.deinit();
}

pub fn generateAndFillRandomColor(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        3 * @sizeOf(u32),
        f32,
        &.{ random.float(f32), random.float(f32), random.float(f32) },
    );
}
pub const updateBuffers = struct {
    stats: Wgpu.ObjectBuffer,
    consumers: Wgpu.ObjectBuffer,
    producers: Wgpu.ObjectBuffer,
};
pub fn update(self: *Self, gctx: *zgpu.GraphicsContext, args: updateBuffers) void {
    const gpu_stats = Wgpu.getAll(gctx, u32, .{
        .structs = args.stats,
        .num_structs = NUM_STATS,
    }) catch unreachable;
    self.num_transactions.append(gpu_stats[0]) catch unreachable;
    clearNumTransactions(gctx, args.stats.data);

    const consumers = Wgpu.getAll(gctx, Consumer, .{
        .structs = args.consumers,
        .num_structs = Wgpu.getNumStructs(gctx, Consumer, args.stats),
    }) catch return;
    var empty_consumers: u32 = 0;
    for (consumers) |c| {
        if (c.inventory == 0) {
            empty_consumers += 1;
        }
    }
    self.num_empty_consumers.append(empty_consumers) catch unreachable;

    const producers = Wgpu.getAll(gctx, Producer, .{
        .structs = args.producers,
        .num_structs = Wgpu.getNumStructs(gctx, Producer, args.stats),
    }) catch return;
    var total_inventory: u32 = 0;
    for (producers) |p| {
        total_inventory += @intCast(u32, p.inventory);
    }
    self.num_total_producer_inventory.append(total_inventory) catch unreachable;
}

pub fn clear(self: *Self) void {
    self.num_transactions.clearAndFree();
    self.num_empty_consumers.clearAndFree();
    self.num_total_producer_inventory.clearAndFree();
    self.num_transactions.append(0) catch unreachable;
    self.num_empty_consumers.append(0) catch unreachable;
    self.num_total_producer_inventory.append(0) catch unreachable;
    self.second = 0;
}

pub fn clearNumTransactions(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, u32, &.{0});
}

pub fn setNumConsumers(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, num: u32) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        @sizeOf(u32),
        u32,
        &.{num},
    );
}

pub fn setNumProducers(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, num: u32) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        2 * @sizeOf(u32),
        u32,
        &.{num},
    );
}

pub fn createBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .storage = true },
        .size = @sizeOf(u32) * zero.len,
    });
}

pub fn createMappedBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = @sizeOf(u32) * zero.len,
    });
}

// pub fn getNumConsumers(gctx: *zgpu.GraphicsContext, stat_bufs: Wgpu.ObjectBuffer) u32 {
//     return Wgpu.getGPUStatistics(gctx, stat_bufs)[1];
// }
// pub fn getNumProducers(gctx: *zgpu.GraphicsContext, stat_bufs: Wgpu.ObjectBuffer) u32 {
//     return getGPUStatistics(gctx, stat_bufs)[2];
// }
