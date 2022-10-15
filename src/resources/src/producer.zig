const std = @import("std");
const array = std.ArrayList;
const zgpu = @import("zgpu");

const Self = @This();

position: @Vector(4, f32),
color: @Vector(4, f32),
production_rate: u32,
giving_rate: u32,
inventory: u32,
max_inventory: u32,
len: u32,
queue: [450]u32,

pub fn createBuffer(gctx: *zgpu.GraphicsContext, producers: array(Self)) zgpu.BufferHandle {
    const max_num_producers = 100;
    const producer_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = max_num_producers * @sizeOf(Self),
    });
    gctx.queue.writeBuffer(gctx.lookupResource(producer_buffer).?, 0, Self, producers.items[0..]);
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
