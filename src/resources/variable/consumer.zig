const std = @import("std");
const math = std.math;
const array = std.ArrayList;
const Allocator = std.mem.Allocator;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Parameters = Main.Parameters;
const Wgpu = @import("wgpu.zig");
const Camera = @import("camera.zig");
const Statistics = @import("statistics.zig");
const Self = @This();

absolute_home: [4]i32 = .{ 0, 0, 0, 0 },
position: [4]f32 = .{ 0, 0, 0, 0 },
home: [4]f32 = .{ 0, 0, 0, 0 },
destination: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 0, 0, 0 },
step_size: [2]f32 = .{ 0, 0 },
radius: f32 = 20.0,
inventory: u32 = 0,
balance: u32 = 0,
max_balance: u32 = 100000,
producer_id: i32 = -1,
grouping_id: u32 = 0,

pub const Params = struct {
    moving_rate: f32 = 0,
    max_demand_rate: u32 = 0,
    income: u32 = 0,
};

pub const z_pos = 0;
pub fn generateBulk(demo: *DemoState, num: u32) void {
    for (0..num) |_| {
        const c = createNewConsumer(demo);
        appendConsumer(demo, c);
    }
}
pub fn createNewConsumer(demo: *DemoState) Self {
    const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
    const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
    const f_x = @as(f32, @floatFromInt(x)) * demo.aspect;
    const f_y = @as(f32, @floatFromInt(y));
    const home = [4]f32{ f_x, f_y, z_pos, 1 };
    return .{
        .absolute_home = .{ x, y, z_pos, 1 },
        .position = home,
        .home = home,
        .destination = home,
    };
}

pub fn appendConsumer(demo: *DemoState, c: Self) void {
    const obj_buf = &demo.buffers.data.consumers;
    var consumers: [1]Self = .{c};
    Wgpu.appendBuffer(demo.gctx, Self, .{
        .num_old_structs = obj_buf.mapping.num_structs,
        .buf = obj_buf.buf,
        .structs = consumers[0..],
    });
    obj_buf.mapping.num_structs += 1;
}

pub fn setParamsBuf(demo: *DemoState, mr: f32, mdr: u32, income: u32) void {
    const r = demo.gctx.lookupResource(demo.buffers.data.consumer_params).?;
    demo.gctx.queue.writeBuffer(r, 0, f32, &.{mr});
    demo.gctx.queue.writeBuffer(r, 4, u32, &.{ mdr, income });
}
