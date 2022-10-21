const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Allocator = std.mem.Allocator;
const zgpu = @import("zgpu");
const Simulation = @import("simulation.zig");

const Self = @This();

position: [4]f32,
color: [4]f32,
production_rate: u32,
giving_rate: u32,
inventory: u32,
max_inventory: u32,
len: u32,
queue: [450]u32,
_padding: u32 = 0,

const max_num_producers = 100;

pub fn createProducers(sim: Simulation) []Self {
    var producers: [max_num_producers]Self = undefined;

    var i: usize = 0;
    while (i < sim.params.num_producers) {
        const x = @intToFloat(
            f32, 
            random.intRangeAtMost(
                i32,
                sim.coordinate_size.min_x,
                sim.coordinate_size.max_x
            )
        );

        const y = @intToFloat(
            f32,
            random.intRangeAtMost(
                i32,
                sim.coordinate_size.min_y,
                sim.coordinate_size.max_y
            )
        );

        producers[i] = Self{
            .position = [4]f32{ x, y, 0, 0 },
            .color = [4]f32{ 1, 1, 1, 0 },
            .production_rate = sim.params.production_rate,
            .giving_rate = sim.params.giving_rate,
            .inventory = sim.params.max_inventory,
            .max_inventory = sim.params.max_inventory,
            .len = 0,
            .queue = [_]u32{0} ** 450,
        };
        i += 1;
    }
    return producers[0..i];
}

//pub fn supplyShock(self: *) void {
//    for (self.producers.items) |_, i| {
//        self.producers.items[i].inventory = 0;
//    }
//}

pub fn createBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    const producer_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = sim.params.num_producers * @sizeOf(Self),
    });

    gctx.queue.writeBuffer(
        gctx.lookupResource(producer_buffer).?,
        0,
        Self,
        createProducers(sim),
    );

    return producer_buffer;
}

pub fn createVertexBuffer(gctx: *zgpu.GraphicsContext, width: f32) zgpu.BufferHandle {
    const producer_vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = 6 * @sizeOf(f32) * 3,
    });

    const upper_left = [3]f32{ -width, width, 0.0 };
    const lower_left = [3]f32{ -width, -width, 0.0 };
    const upper_right = [3]f32{ width, width, 0.0 };
    const lower_right = [3]f32{ width, -width, 0.0 };

    const vertex_array = [6][3]f32{
        upper_left,
        lower_left,
        lower_right,
        lower_right,
        upper_right,
        upper_left,
    };

    gctx.queue.writeBuffer(gctx.lookupResource(producer_vertex_buffer).?, 0, [3]f32, vertex_array[0..]);
    return producer_vertex_buffer;
}
