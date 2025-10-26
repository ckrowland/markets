const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const Camera = @import("camera");
const Wgpu = @import("wgpu");
const Self = @This();

absolute_home: [4]f32 = .{ 0, 0, 0, 0 },
home: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 1, 1, 0 },
inventory: u32 = 5000,
max_inventory: u32,
money: u32 = 5000,
max_money: u32 = 10000,
price: u32,
production_cost: u32,
_padding1: u32 = 0,
_padding2: u32 = 0,

pub const z_pos = 0;
pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    obj_buf: *Wgpu.ObjectBuffer(Self),
    num: u32,
    p: Self,
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
            .home = world_pos,
            .max_inventory = p.max_inventory,
            .production_cost = p.production_cost,
            .price = p.price,
        });
    }
}
