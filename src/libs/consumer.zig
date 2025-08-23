const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const Camera = @import("camera");
const Self = @This();

absolute_home: [4]i32 = .{ 0, 0, 0, 0 },
position: [4]f32 = .{ 0, 0, 0, 0 },
home: [4]f32 = .{ 0, 0, 0, 0 },
destination: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 0, 0, 0 },
step_size: [2]f32 = .{ 0, 0 },
inventory: u32 = 0,
radius: f32 = 20.0,
producer_id: i32 = -1,
grouping_id: u32 = 0,
_padding0: u32 = 0,
_padding1: u32 = 0,

pub const Params = struct {
    moving_rate: f32 = 0,
    demand_rate: i32 = 0,
};

pub const z_pos = 0;
pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    num_structs: *u32,
    aspect: f32,
    num: u32,
) void {
    for (0..num) |_| {
        const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
        const f_x = @as(f32, @floatFromInt(x)) * aspect;
        const f_y = @as(f32, @floatFromInt(y));
        const home = [4]f32{ f_x, f_y, z_pos, 1 };
        var consumers: [1]Self = .{
            .{
                .absolute_home = .{ x, y, z_pos, 1 },
                .position = home,
                .home = home,
                .destination = home,
            },
        };

        const resource = gctx.lookupResource(buf).?;
        gctx.queue.writeBuffer(
            resource,
            num_structs.* * @sizeOf(Self),
            Self,
            consumers[0..],
        );
        num_structs.* += 1;
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
        .radius = consumer.radius,
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

pub fn setParamsBuf(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    mr: f32,
    mdr: u32,
) void {
    const r = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(r, 0, f32, &.{mr});
    gctx.queue.writeBuffer(r, 4, u32, &.{mdr});
}
