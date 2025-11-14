const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const Camera = @import("camera.zig");
const Wgpu = @import("wgpu.zig");
const Self = @This();

absolute_home: [4]f32 = .{ 0, 0, 0, 0 },
position: [4]f32 = .{ 0, 0, 0, 0 },
home: [4]f32 = .{ 0, 0, 0, 0 },
destination: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 0, 0, 0 },
step_size: [2]f32 = .{ 0, 0 },
inventory: u32 = 0,
money: u32 = 0,
max_money: u32,
income: u32,
moving_rate: f32,
producer_id: u32 = 0,
grouping_id: u32 = 0,
_padding0: u32 = 0,
_padding1: u32 = 0,
_padding2: u32 = 0,

pub const z_pos = 0;
pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    obj_buf: *Wgpu.ObjectBuffer(Self),
    num: u32,
    params: Self,
) void {
    for (0..num) |_| {
        const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
        const a_x = @as(f32, @floatFromInt(x));
        const f_x = @as(f32, @floatFromInt(x)) * Camera.getAspectRatio(gctx);
        const f_y = @as(f32, @floatFromInt(y));
        const grid_pos = [4]f32{ a_x, f_y, z_pos, 1 };
        const world_pos = [4]f32{ f_x, f_y, z_pos, 1 };
        obj_buf.append(gctx, .{
            .absolute_home = grid_pos,
            .position = world_pos,
            .home = world_pos,
            .destination = world_pos,
            .moving_rate = params.moving_rate,
            .income = params.income,
            .max_money = params.max_money,
        });
    }
}

pub fn createAndAppend(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    num_structs: *u32,
    consumer: Self,
) void {
    const home: [4]f32 = .{ consumer.home[0], consumer.home[1], z_pos, 1 };
    const absolute_home: [4]i32 = .{
        consumer.absolute_home[0],
        consumer.absolute_home[1],
        z_pos,
        1,
    };
    const c = Self{
        .absolute_home = absolute_home,
        .position = home,
        .home = home,
        .destination = home,
        .color = consumer.color,
        .grouping_id = consumer.grouping_id,
    };
    const resource = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(
        resource,
        num_structs.* * @sizeOf(Self),
        Self,
        &.{c},
    );
    num_structs.* += 1;
}
