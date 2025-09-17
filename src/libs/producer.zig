const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const Camera = @import("camera");
const Wgpu = @import("wgpu");

const Self = @This();

absolute_home: [4]i32 = .{ 0, 0, 0, 0 },
home: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 1, 1, 0 },
inventory: i32 = 5000,
max_inventory: i32,
money: i32 = 5000,
max_money: i32 = 10000,
price: i32,
production_cost: i32,
_padding1: u32 = 0,
_padding2: u32 = 0,

pub const z_pos = 0;
pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    num_structs: *u32,
    aspect: f32,
    num: u32,
    p: Self,
) void {
    for (0..num) |_| {
        const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
        const f_x = @as(f32, @floatFromInt(x)) * aspect;
        const f_y = @as(f32, @floatFromInt(y));
        const home = [4]f32{ f_x, f_y, z_pos, 1 };
        createAndAppend(gctx, buf, num_structs, .{
            .absolute_home = .{ x, y, z_pos, 1 },
            .home = home,
            .max_inventory = p.max_inventory,
            .production_cost = p.production_cost,
            .price = p.price,
        });
    }
}

pub fn createAndAppend(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    num_structs: *u32,
    producer: Self,
) void {
    const resource = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(
        resource,
        num_structs.* * @sizeOf(Self),
        Self,
        &.{producer},
    );
    num_structs.* += 1;
}
