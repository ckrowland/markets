const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const Camera = @import("camera.zig");
const Wgpu = @import("wgpu.zig");
const Self = @This();

home: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 1, 1, 0 },
inventory: u32 = 5000,
max_inventory: u32,
money: u32 = 5000,
max_money: u32,
price: u32,
production_cost: u32,
max_production_rate: u32,
decay_rate: u32,

pub const z_pos = 0;
pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    obj_buf: *Wgpu.ObjectBuffer(Self),
    num: u32,
    p: Self,
) void {
    for (0..num) |_| {
        const x = random.intRangeAtMost(u32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(u32, Camera.MIN_Y, Camera.MAX_Y);
        const f_x = @as(f32, @floatFromInt(x));
        const f_y = @as(f32, @floatFromInt(y));
        const world_pos = [4]f32{ f_x, f_y, z_pos, 1 };
        obj_buf.append(gctx, .{
            .home = world_pos,
            .max_inventory = p.max_inventory,
            .production_cost = p.production_cost,
            .max_production_rate = p.max_production_rate,
            .price = p.price,
            .max_money = p.max_money,
            .decay_rate = p.decay_rate,
        });
    }
}
