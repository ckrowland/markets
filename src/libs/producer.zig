const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const Camera = @import("camera");

const Self = @This();

absolute_home: [4]i32,
home: [4]f32,
color: [4]f32 = .{ 1, 1, 1, 0 },
production_rate: u32 = 300,
inventory: u32 = 0,
available_inventory: u32 = 0,
max_inventory: u32 = 10000,

pub const z_pos = 0;
pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    num_structs: *u32,
    aspect: f32,
    num: u32,
    pr: u32,
    mi: u32,
) void {
    for (0..num) |_| {
        const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
        const f_x = @as(f32, @floatFromInt(x)) * aspect;
        const f_y = @as(f32, @floatFromInt(y));
        const home = [4]f32{ f_x, f_y, z_pos, 1 };
        const p = Self{
            .absolute_home = .{ x, y, z_pos, 1 },
            .home = home,
            .production_rate = pr,
            .max_inventory = mi,
        };
        createAndAppend(gctx, buf, num_structs, p);
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
