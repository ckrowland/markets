const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Allocator = std.mem.Allocator;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu.zig");
const DemoState = @import("random/main.zig");
const Parameters = DemoState.Parameters;
const CoordinateSize = DemoState.CoordinateSize;
const Camera = @import("../camera.zig");

const Self = @This();

absolute_home: [4]i32,
home: [4]f32,
color: [4]f32,
production_rate: u32,
inventory: i32,
max_inventory: u32,
_padding1: u32 = 0,

pub const z_pos = 0;
pub const Parameter = enum {
    production_rate,
    supply_shock,
    max_inventory,
};

pub const Args = struct {
    absolute_home: [2]i32,
    home: [2]f32,
    color: [4]f32 = .{ 1, 1, 1, 0 },
    production_rate: u32 = 300,
    inventory: i32 = 0,
    max_inventory: u32 = 10000,
};
pub fn create(args: Args) Self {
    return Self{
        .absolute_home = .{ args.absolute_home[0], args.absolute_home[1], z_pos, 1 },
        .home = .{ args.home[0], args.home[1], z_pos, 1 },
        .color = args.color,
        .production_rate = args.production_rate,
        .inventory = args.inventory,
        .max_inventory = args.max_inventory,
    };
}

pub fn generateBulk(
    gctx: *zgpu.GraphicsContext,
    obj_buf: Wgpu.ObjectBuffer,
    params: Parameters,
) void {
    var producers: [DemoState.MAX_NUM_PRODUCERS]Self = undefined;
    const p_len = createBulk(&producers, params, params.num_producers.new);
    gctx.queue.writeBuffer(
        gctx.lookupResource(obj_buf.data).?,
        0,
        Self,
        producers[0..p_len],
    );
    Wgpu.writeToMappedBuffer(gctx, obj_buf);
}

pub fn createBulk(slice: []Self, params: Parameters, num: usize) usize {
    var producers: [DemoState.MAX_NUM_PRODUCERS]Self = undefined;
    var i: usize = 0;
    while (i < num) {
        const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
        const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
        producers[i] = create(.{
            .absolute_home = .{ x, y },
            .home = [2]f32{ @as(f32, @floatFromInt(x)) * params.aspect.*, @as(f32, @floatFromInt(y)) },
            .production_rate = params.production_rate,
            .inventory = @as(i32, @intCast(params.max_inventory)),
            .max_inventory = params.max_inventory,
        });
        i += 1;
    }
    std.mem.copy(Self, slice, &producers);
    return i;
}

pub const updateCoordsArgs = struct {
    producers: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
};
pub fn updateCoords(gctx: *zgpu.GraphicsContext, args: updateCoordsArgs) void {
    const producers = Wgpu.getAll(gctx, Self, .{
        .structs = args.producers,
        .num_structs = Wgpu.getNumStructs(gctx, Self, args.stats),
    }) catch return;
    var new_producers: [DemoState.MAX_NUM_PRODUCERS]Self = undefined;
    for (producers, 0..) |p, i| {
        new_producers[i] = p;
        new_producers[i].home = Camera.getWorldPosition(gctx, p.absolute_home);
    }
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.producers.data).?,
        0,
        Self,
        new_producers[0..producers.len],
    );
}
