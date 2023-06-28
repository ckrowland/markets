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

pub const defaults = DEFAULTS{};
const DEFAULTS = struct {
    color: [4]f32 = .{ 1, 0, 0, 0 },
    moving_rate: f32 = 5.0,
    demand_rate: u32 = 100,
    radius: f32 = 20.0,
};
    
absolute_home: [4]f32,
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
pub fn generateBulk(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, params: Parameters) void {
    var consumers: [DemoState.MAX_NUM_CONSUMERS]Self = undefined;
    const c_len = createRandomBulk(&consumers, params, params.num_consumers.new);
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        Self,
        consumers[0..c_len],
    );
}

pub fn createRandomBulk(slice: []Self, params: Parameters, num: u32) usize {
    var consumers: [DemoState.MAX_NUM_CONSUMERS]Self = undefined;
    var i: usize = 0;
    while (i < num) {
        const x = @floatFromInt(f32, random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X));
        const y = @floatFromInt(f32, random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y));
        const aspect_home = [2]f32{ x * params.aspect, y };

        consumers[i] = create(.{
            .absolute_home = .{ x, y },
            .home = aspect_home,
            .moving_rate = params.moving_rate,
            .demand_rate = params.demand_rate,
            .radius = params.consumer_radius,
        });
        i += 1;
    }
    std.mem.copy(Self, slice, &consumers);
    return i;
}

pub const Args = struct {
    absolute_home: [2]f32,
    home: [2]f32,
    color: [4]f32 = defaults.color,
    moving_rate: f32 = defaults.moving_rate,
    demand_rate: u32 = defaults.demand_rate,
    radius: f32 = defaults.radius,
    grouping_id: u32 = 0,
};
pub fn create(args: Args) Self {
    const home: [4]f32 = .{ args.home[0], args.home[1], z_pos, 1 };
    return Self{
        .absolute_home = .{args.absolute_home[0], args.absolute_home[1], z_pos, 1 },
        .position = home,
        .home = home,
        .destination = home,
        .color = args.color,
        .moving_rate = args.moving_rate,
        .demand_rate = args.demand_rate,
        .radius = args.radius,
        .grouping_id = args.grouping_id,
    };
}

pub const updateCoordsArgs = struct {
    consumers: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
};

pub fn updateCoords(gctx: *zgpu.GraphicsContext, args: updateCoordsArgs) void {
    const consumers = Wgpu.getAll(gctx, Self, .{
        .structs = args.consumers,
        .num_structs = Wgpu.getNumStructs(gctx, Self, args.stats),
    }) catch return;
    var new_consumers: [DemoState.MAX_NUM_CONSUMERS]Self = undefined;
    for (consumers, 0..) |c, i| {
        const world_pos = Camera.getWorldPosition(gctx, c.absolute_home);
        new_consumers[i] = c;
        new_consumers[i].position = world_pos;
        new_consumers[i].home = world_pos;
        new_consumers[i].destination = world_pos;
    }
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.consumers.data).?,
        0,
        Self,
        new_consumers[0..consumers.len],
    );
}
