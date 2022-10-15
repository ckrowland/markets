const std = @import("std");
const math = std.math;
const array = std.ArrayList;
const zgpu = @import("zgpu");

const Self = @This();

position: @Vector(4, f32),
home: @Vector(4, f32),
destination: @Vector(4, f32),
step_size: @Vector(4, f32),
color: @Vector(4, f32),
moving_rate: f32,
inventory: u32,
radius: f32,
producer_id: i32,

const num_vertices = 20;

pub fn createBuffer(gctx: *zgpu.GraphicsContext, consumers: array(Self)) zgpu.BufferHandle {
    const max_num_consumer = 10000;
    const consumer_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = max_num_consumer * @sizeOf(Self),
    });
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_buffer).?, 0, Self, consumers.items[0..]);
    return consumer_buffer;
}

pub fn createIndexBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const num_triangles = num_vertices - 1;
    const consumer_index_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = num_triangles * 3 * @sizeOf(u32),
    });
    const consumer_index_data = createIndexData(num_triangles);
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_index_buffer).?, 0, i32, consumer_index_data[0..]);
    return consumer_index_buffer;
}

pub fn createVertexBuffer(gctx: *zgpu.GraphicsContext, radius: f32) zgpu.BufferHandle {
    const consumer_vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = num_vertices * @sizeOf(f32) * 3,
    });
    var consumer_vertex_data: [num_vertices][3]f32 = undefined;
    const num_sides = @as(f32, num_vertices - 1);
    const angle = 2 * math.pi / num_sides;

    consumer_vertex_data[0] = [3]f32{ 0, 0, 0, };
    var i: u32 = 1;
    while (i < num_vertices) {
        const current_angle = angle * @intToFloat(f32, i);
        const x = @cos(current_angle) * radius;
        const y = @sin(current_angle) * radius;
        consumer_vertex_data[i] = [3]f32{ x, y, 0 };
        i += 1;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_vertex_buffer).?, 0, [3]f32, consumer_vertex_data[0..]);
    return consumer_vertex_buffer;
}

pub fn createIndexData(comptime num_triangles: u32) [num_triangles * 3]i32 {
    const vertices = num_triangles * 3;
    var consumer_index_data: [vertices]i32 = undefined;
    var i: usize = 0;
    while (i < num_triangles) {
        const idx = i * 3;
        const triangle_num = @intCast(i32, i);
        consumer_index_data[idx] = 0;
        consumer_index_data[idx + 1] = triangle_num + 1;
        consumer_index_data[idx + 2] = triangle_num + 2;
        i += 1;
    }
    consumer_index_data[vertices - 1] = 1;
    return consumer_index_data;
}
