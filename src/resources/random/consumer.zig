const std = @import("std");
const math = std.math;
const array = std.ArrayList;
const Allocator = std.mem.Allocator;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const DemoState = @import("main.zig");
const Parameters = DemoState.Parameters;
const Wgpu = @import("wgpu.zig");
const Camera = @import("camera.zig");
const Statistics = @import("statistics.zig");
const Self = @This();

pub const defaults = DEFAULTS{};
const DEFAULTS = struct {
    color: [4]f32 = .{ 1, 0, 0, 0 },
    moving_rate: f32 = 5.0,
    demand_rate: u32 = 100,
    radius: f32 = 20.0,
};

absolute_home: [4]i32,
position: [4]f32,
home: [4]f32,
destination: [4]f32,
color: [4]f32 = defaults.color,
step_size: [2]f32 = .{ 0, 0 },
moving_rate: f32,
demand_rate: u32,
inventory: u32 = 0,
radius: f32,
producer_id: i32 = -1,
grouping_id: u32 = 0,

pub const z_pos = 0;
pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    obj_buf: *Wgpu.ObjectBuffer(Self),
    params: Parameters,
    num: u32,
) void {
    var i: usize = 0;
    while (i < num) {
        const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
        const aspect_home = [2]f32{
            @as(f32, @floatFromInt(x)) * params.aspect,
            @as(f32, @floatFromInt(y)),
        };

        createAndAppend(gctx, .{
            .consumer = .{
                .absolute_home = .{ x, y },
                .home = aspect_home,
                .moving_rate = params.moving_rate,
                .demand_rate = params.demand_rate,
                .radius = params.consumer_radius,
            },
            .obj_buf = obj_buf,
        });
        i += 1;
    }
}

pub const Args = struct {
    absolute_home: [2]i32,
    home: [2]f32,
    color: [4]f32 = defaults.color,
    moving_rate: f32 = defaults.moving_rate,
    demand_rate: u32 = defaults.demand_rate,
    radius: f32 = defaults.radius,
    grouping_id: u32 = 0,
};
pub const AppendArgs = struct {
    consumer: Args,
    obj_buf: *Wgpu.ObjectBuffer(Self),
};
pub fn createAndAppend(gctx: *zgpu.GraphicsContext, args: AppendArgs) void {
    const home: [4]f32 = .{
        args.consumer.home[0],
        args.consumer.home[1],
        z_pos,
        1,
    };
    const absolute_home: [4]i32 = .{
        args.consumer.absolute_home[0],
        args.consumer.absolute_home[1],
        z_pos,
        1,
    };
    const consumer = Self{
        .absolute_home = absolute_home,
        .position = home,
        .home = home,
        .destination = home,
        .color = args.consumer.color,
        .moving_rate = args.consumer.moving_rate,
        .demand_rate = args.consumer.demand_rate,
        .radius = args.consumer.radius,
        .grouping_id = args.consumer.grouping_id,
    };
    var consumers: [1]Self = .{consumer};
    Wgpu.appendBuffer(gctx, Self, .{
        .num_old_structs = @as(u32, @intCast(args.obj_buf.list.items.len)),
        .buf = args.obj_buf.buf,
        .structs = consumers[0..],
    });
    args.obj_buf.list.append(consumers[0]) catch unreachable;
    args.obj_buf.mapping.num_structs += 1;
}
