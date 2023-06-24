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

absolute_home: [4]f32,
home: [4]f32,
color: [4]f32 = .{ 1, 1, 1, 0 },
production_rate: u32 = 300,
inventory: i32 = 0,
max_inventory: u32 = 10000,
_padding1: u32 = 0,

pub const Parameter = enum {
    production_rate,
    supply_shock,
    max_inventory,
};

pub const Args = struct {
    absolute_home: [4]f32,
    home: [4]f32,
};
pub fn create(args: Args) Self {
    return Self{
        .absolute_home = args.absolute_home,
        .home = args.home,
    };
}

pub fn generateBulk(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, params: Parameters) void {
    var producers: [DemoState.MAX_NUM_PRODUCERS]Self = undefined;
    const p_len = createBulk(&producers, params, params.num_producers.new);
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        Self,
        producers[0..p_len],
    );
}

pub fn createBulk(slice: []Self, params: Parameters, num: usize) usize {
    var producers: [DemoState.MAX_NUM_PRODUCERS]Self = undefined;
    var i: usize = 0;
    while (i < num) {
        const x = @intToFloat(f32, random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X));
        const y = @intToFloat(f32, random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y));
        producers[i] = Self{
            .absolute_home = [4]f32{ x, y, 0, 0 },
            .home = [4]f32{ x * params.aspect, y, 0, 0 },
            .production_rate = params.production_rate,
            .inventory = @intCast(i32, params.max_inventory),
            .max_inventory = params.max_inventory,
        };
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
