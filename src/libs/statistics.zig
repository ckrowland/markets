const std = @import("std");
const zgpu = @import("zgpu");
const Wgpu = @import("wgpu");
const Self = @This();

num_transactions: std.ArrayList(u32),
second: f32 = 0,
num_empty_consumers: std.ArrayList(u32),
num_total_producer_inventory: std.ArrayList(u32),
obj_buf: Wgpu.ObjectBuffer(u32),

pub const NUM_STATS = 8;
pub const zero = [NUM_STATS]u32{ 0, 0, 0, 0, 0, 0, 0, 0 };

pub fn init(gctx: *zgpu.GraphicsContext, allocator: std.mem.Allocator) Self {
    const stats_object = Wgpu.createObjectBuffer(
        gctx,
        u32,
        NUM_STATS,
        NUM_STATS,
    );
    return Self{
        .num_transactions = std.ArrayList(u32).init(allocator),
        .num_empty_consumers = std.ArrayList(u32).init(allocator),
        .num_total_producer_inventory = std.ArrayList(u32).init(allocator),
        .obj_buf = stats_object,
    };
}

pub fn deinit(self: *Self) void {
    self.num_transactions.deinit();
    self.num_empty_consumers.deinit();
    self.num_total_producer_inventory.deinit();
}

pub fn clear(self: *Self) void {
    self.num_transactions.clearAndFree();
    self.num_empty_consumers.clearAndFree();
    self.num_total_producer_inventory.clearAndFree();
}

pub const setParam = enum(u32) {
    num_transactions = 0,
    consumers = 1,
    producers = 2,
    consumer_hovers = 3,
};
pub fn setNum(
    self: *Self,
    gctx: *zgpu.GraphicsContext,
    num: u32,
    param: setParam,
) void {
    const resource = gctx.lookupResource(self.obj_buf.buf).?;
    const offset = @intFromEnum(param) * @sizeOf(u32);
    gctx.queue.writeBuffer(resource, offset, u32, &.{num});
}
