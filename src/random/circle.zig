const std = @import("std");
const math = std.math;
const zgpu = @import("zgpu");

const Self = @This();

position: [4]f32,
color: [4]f32,
radius: f32,

pub fn createIndexBuffer(gctx: *zgpu.GraphicsContext, comptime num_vertices: u32) zgpu.BufferHandle {
    const num_triangles = num_vertices - 1;
    const consumer_index_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = num_triangles * 3 * @sizeOf(u32),
    });

    const num_indices = num_triangles * 3;
    var indices: [num_indices]u32 = undefined;
    var i: usize = 0;
    while (i < num_triangles) {
        indices[i * 3] = 0;
        indices[i * 3 + 1] = @as(u32, @intCast(i)) + 1;
        indices[i * 3 + 2] = @as(u32, @intCast(i)) + 2;
        i += 1;
    }
    indices[num_indices - 1] = 1;
    const data = indices[0..num_indices];

    gctx.queue.writeBuffer(gctx.lookupResource(consumer_index_buffer).?, 0, u32, data);
    return consumer_index_buffer;
}

pub fn createVertexBuffer(
    gctx: *zgpu.GraphicsContext,
    comptime num_vertices: u32,
    radius: f32,
) zgpu.BufferHandle {
    const consumer_vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = num_vertices * @sizeOf(f32) * 3,
    });
    var consumer_vertex_data: [num_vertices][3]f32 = undefined;
    const num_sides = @as(f32, num_vertices - 1);
    const angle = 2 * math.pi / num_sides;
    consumer_vertex_data[0] = [3]f32{ 0, 0, 0 };
    var i: u32 = 1;
    while (i < num_vertices) {
        const current_angle = angle * @as(f32, @floatFromInt(i));
        const x = @cos(current_angle) * radius;
        const y = @sin(current_angle) * radius;
        consumer_vertex_data[i] = [3]f32{ x, y, 0 };
        i += 1;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_vertex_buffer).?, 0, [3]f32, consumer_vertex_data[0..]);
    return consumer_vertex_buffer;
}
