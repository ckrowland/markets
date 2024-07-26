const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const Wgpu = @import("wgpu.zig");
const Self = @This();

num_transactions: array(u32),
second: f32 = 0,
num_empty_consumers: array(u32),
num_total_producer_inventory: array(u32),

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
        4 * @sizeOf(u32),
        f32,
        &.{ random.float(f32), random.float(f32), random.float(f32) },
    );
}

pub fn clear(self: *Self) void {
    self.num_transactions.clearAndFree();
    self.num_empty_consumers.clearAndFree();
    self.num_total_producer_inventory.clearAndFree();
}

pub fn clearNumTransactions(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, u32, &.{0});
}

pub const setArgs = struct {
    stat_obj: Wgpu.ObjectBuffer(u32),
    num: u32,
    param: enum(u32) {
        num_transactions = 0,
        consumers = 1,
        producers = 2,
        consumer_hovers = 3,
    },
};
pub fn setNum(gctx: *zgpu.GraphicsContext, args: setArgs) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.stat_obj.buf).?,
        @intFromEnum(args.param) * @sizeOf(u32),
        u32,
        &.{args.num},
    );
}
