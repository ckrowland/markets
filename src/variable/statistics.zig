const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const DemoState = @import("main.zig").DemoState;
const Wgpu = @import("wgpu.zig");
const Self = @This();

num_transactions: array(u32),
second: f32 = 0,
num_empty_consumers: array(u32),
num_total_producer_inventory: array(u32),
avg_consumer_balance: array(u32),
avg_producer_balance: array(u32),
avg_margin: array(u32),

pub const NUM_STATS = 8;
pub const zero = [NUM_STATS]u32{ 0, 0, 0, 0, 0, 0, 0, 0 };

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .num_transactions = array(u32).init(allocator),
        .num_empty_consumers = array(u32).init(allocator),
        .num_total_producer_inventory = array(u32).init(allocator),
        .avg_consumer_balance = array(u32).init(allocator),
        .avg_producer_balance = array(u32).init(allocator),
        .avg_margin = array(u32).init(allocator),
    };
}

pub fn deinit(self: *Self) void {
    self.num_transactions.deinit();
    self.num_empty_consumers.deinit();
    self.num_total_producer_inventory.deinit();
    self.avg_consumer_balance.deinit();
    self.avg_producer_balance.deinit();
    self.avg_margin.deinit();
}

pub fn generateAndFillRandomColor(demo: *DemoState) void {
    const handle = demo.buffers.data.stats.buf;
    const resource = demo.gctx.lookupResource(handle).?;
    const color = [3]f32{ random.float(f32), random.float(f32), random.float(f32) };
    demo.gctx.queue.writeBuffer(resource, 3 * @sizeOf(u32), [3]f32, &.{color});
}

pub fn clear(self: *Self) void {
    self.num_transactions.clearAndFree();
    self.num_empty_consumers.clearAndFree();
    self.num_total_producer_inventory.clearAndFree();
    self.avg_consumer_balance.clearAndFree();
    self.avg_producer_balance.clearAndFree();
    self.avg_margin.clearAndFree();
}

pub fn clearNumTransactions(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, u32, &.{0});
}

pub const Param = enum(u32) {
    num_transactions = 0,
    consumers = 1,
    producers = 2,
    consumer_hovers = 3,
};
pub fn setNum(demo: *DemoState, num: u32, param: Param) void {
    const resource = demo.gctx.lookupResource(demo.buffers.data.stats.buf).?;
    const offset = @intFromEnum(param) * @sizeOf(u32);
    demo.gctx.queue.writeBuffer(resource, offset, u32, &.{num});
}
