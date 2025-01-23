const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Allocator = std.mem.Allocator;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu");
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Parameters = Main.Parameters;
const Camera = @import("camera");

pub const Self = @This();

absolute_home: [4]i32 = .{ 0, 0, 0, 0 },
home: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 1, 1, 0 },
production_rate: u32,
inventory: i32 = 0,
max_inventory: u32 = 10000,
_padding0: u32 = 0,

pub const z_pos = 0;
pub const Parameter = enum {
    production_rate,
    supply_shock,
    max_inventory,
};

pub fn generateBulk(demo: *DemoState, num: u32) void {
    for (0..num) |_| {
        const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);

        var producers: [1]Self = .{
            Self{
                .absolute_home = .{ x, y, z_pos, 1 },
                .home = .{
                    @as(f32, @floatFromInt(x)) * demo.aspect,
                    @as(f32, @floatFromInt(y)),
                    z_pos,
                    1,
                },
                .color = .{ 1, 1, 1, 0 },
                .max_inventory = demo.sliders.get("max_inventory").?.slider.val,
                .inventory = @intCast(demo.sliders.get("max_inventory").?.slider.val),
                .production_rate = @intCast(demo.sliders.get("production_rate").?.slider.val),
            },
        };
        Wgpu.appendBuffer(demo.gctx, Self, .{
            .num_old_structs = demo.buffers.data.producers.mapping.num_structs,
            .buf = demo.buffers.data.producers.buf,
            .structs = producers[0..],
        });
        demo.buffers.data.producers.mapping.num_structs += 1;
    }
}
