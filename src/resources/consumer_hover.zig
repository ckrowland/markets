const std = @import("std");
const math = std.math;
const array = std.ArrayList;
const Allocator = std.mem.Allocator;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const DemoState = @import("random/main.zig");
const Parameters = DemoState.Parameters;
const CoordinateSize = DemoState.CoordinateSize;
const Wgpu = @import("wgpu.zig");
const Camera = @import("../camera.zig");
const Self = @This();

absolute_home: [4]f32,
home: [4]f32,
color: [4]f32 = .{ 0, 1, 1, 0 },
radius: f32 = 60.0,
_padding1: u32 = 0,
_padding2: u32 = 0,
_padding3: u32 = 0,

pub const z_pos = -5;

pub const Args = struct {
    absolute_home: [2]f32,
    home: [2]f32,
};
pub fn create(args: Args) Self {
    return Self{
        .absolute_home = .{ args.absolute_home[0], args.absolute_home[1], z_pos, 1 },
        .home = .{ args.home[0], args.home[1], z_pos, 1 },
    };
}

pub const updateCoordsArgs = struct {
    consumer_hover: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
};
pub fn updateCoords(gctx: *zgpu.GraphicsContext, args: updateCoordsArgs) void {
    const consumers = Wgpu.getAll(gctx, Self, .{
        .structs = args.consumer_hover,
        .num_structs = Wgpu.getNumStructs(gctx, Self, args.stats),
    }) catch return;
    var new_consumers: [DemoState.MAX_NUM_CONSUMERS]Self = undefined;
    for (consumers, 0..) |c, i| {
        const world_pos = Camera.getWorldPosition(gctx, c.absolute_home);
        new_consumers[i] = c;
        new_consumers[i].home = world_pos;
    }
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.consumer_hover).?,
        0,
        Self,
        new_consumers[0..consumers.len],
    );
}
