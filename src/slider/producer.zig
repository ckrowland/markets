const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Allocator = std.mem.Allocator;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu");
const DemoState = @import("main.zig");
const Parameters = DemoState.Parameters;
const Camera = @import("camera");

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

pub const DEFAULT_PRODUCTION_RATE: u32 = 300;
pub const DEFAULT_MAX_INVENTORY: u32 = 10000;
pub const Args = struct {
    absolute_home: [2]i32,
    home: [2]f32,
    color: [4]f32 = .{ 1, 1, 1, 0 },
    production_rate: u32 = DEFAULT_PRODUCTION_RATE,
    inventory: i32 = 0,
    max_inventory: u32 = DEFAULT_MAX_INVENTORY,
};

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
        createAndAppend(gctx, .{
            .obj_buf = obj_buf,
            .producer = .{
                .absolute_home = .{ x, y },
                .home = [2]f32{ @as(f32, @floatFromInt(x)) * params.aspect, @as(f32, @floatFromInt(y)) },
                .production_rate = params.production_rate,
                .inventory = @as(i32, @intCast(params.max_inventory)),
                .max_inventory = params.max_inventory,
            },
        });
        i += 1;
    }
    // Wgpu.writeToMappedBuffer(gctx, obj_buf.buf, obj_buf.mapping.buf);
}

pub const AppendArgs = struct {
    producer: Args,
    obj_buf: *Wgpu.ObjectBuffer(Self),
};
pub fn createAndAppend(gctx: *zgpu.GraphicsContext, args: AppendArgs) void {
    const home: [4]f32 = .{
        args.producer.home[0],
        args.producer.home[1],
        z_pos,
        1,
    };
    const absolute_home: [4]i32 = .{
        args.producer.absolute_home[0],
        args.producer.absolute_home[1],
        z_pos,
        1,
    };
    const producer = Self{
        .absolute_home = absolute_home,
        .home = home,
        .color = args.producer.color,
        .production_rate = args.producer.production_rate,
        .inventory = args.producer.inventory,
        .max_inventory = args.producer.max_inventory,
    };
    var producers: [1]Self = .{producer};
    Wgpu.appendBuffer(gctx, Self, .{
        .num_old_structs = args.obj_buf.mapping.num_structs,
        .buf = args.obj_buf.buf,
        .structs = producers[0..],
    });
    args.obj_buf.mapping.num_structs += 1;
}

// pub const updateCoordsArgs = struct {
//     producers: Wgpu.ObjectBuffer,
//     stats: Wgpu.ObjectBuffer,
// };
// pub fn updateCoords(gctx: *zgpu.GraphicsContext, args: updateCoordsArgs) void {
//     const producers = Wgpu.getAll(gctx, Self, .{
//         .structs = args.producers,
//         .num_structs = Wgpu.getNumStructs(gctx, Self, args.stats),
//     }) catch return;
//     var new_producers: [DemoState.MAX_NUM_PRODUCERS]Self = undefined;
//     for (producers, 0..) |p, i| {
//         new_producers[i] = p;
//         new_producers[i].home = Camera.getWorldPosition(gctx, p.absolute_home);
//     }
//     gctx.queue.writeBuffer(
//         gctx.lookupResource(args.producers.data).?,
//         0,
//         Self,
//         new_producers[0..producers.len],
//     );
// }
