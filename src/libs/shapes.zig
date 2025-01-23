const std = @import("std");
const math = std.math;
const zgpu = @import("zgpu");

pub fn createSquareVertexBuffer(gctx: *zgpu.GraphicsContext, width: f32) zgpu.BufferHandle {
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

pub fn createCircleIndexBuffer(gctx: *zgpu.GraphicsContext, comptime num_vertices: u32) zgpu.BufferHandle {
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

    gctx.queue.writeBuffer(gctx.lookupResource(consumer_index_buffer).?, 0, u32, indices[0..]);
    return consumer_index_buffer;
}

pub fn createCircleVertexBuffer(
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
