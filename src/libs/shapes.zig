const std = @import("std");
const zgpu = @import("zgpu");

pub fn createSquareVertexBuffer(gctx: *zgpu.GraphicsContext, width: f32) zgpu.BufferHandle {
    const buf = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = 4 * @sizeOf(f32) * 3,
    });

    const upper_left = [3]f32{ -width, width, 0.0 };
    const lower_left = [3]f32{ -width, -width, 0.0 };
    const upper_right = [3]f32{ width, width, 0.0 };
    const lower_right = [3]f32{ width, -width, 0.0 };

    const vertex_array = [4][3]f32{
        upper_left,
        lower_left,
        lower_right,
        upper_right,
    };

    const resource = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(resource, 0, [3]f32, vertex_array[0..]);
    return buf;
}

pub fn createSquareIndexBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const buf = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = 6 * @sizeOf(u32),
    });

    var indices: [6]u32 = .{ 0, 1, 2, 2, 3, 0 };
    const resource = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(resource, 0, u32, indices[0..]);
    return buf;
}

pub fn createMoneySquareIndexBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const buf = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = 8 * @sizeOf(u32),
    });

    var indices: [8]u32 = .{ 0, 1, 1, 2, 2, 3, 3, 0 };
    const resource = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(resource, 0, u32, indices[0..]);
    return buf;
}

pub fn createMoneyCircleIndexBuffer(gctx: *zgpu.GraphicsContext, comptime num_sides: u32) zgpu.BufferHandle {
    const buf = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = num_sides * 2 * @sizeOf(u32),
    });

    const num_indices = num_sides * 2;
    var indices: [num_indices]u32 = undefined;
    for (0..num_sides) |i| {
        indices[i * 2] = @as(u32, @intCast(i)) + 1;
        indices[i * 2 + 1] = @as(u32, @intCast(i)) + 2;
    }
    indices[num_indices - 1] = 1;
    const resource = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(resource, 0, u32, indices[0..]);
    return buf;
}

pub fn createCircleIndexBuffer(gctx: *zgpu.GraphicsContext, comptime num_sides: u32) zgpu.BufferHandle {
    const consumer_index_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = num_sides * 3 * @sizeOf(u32),
    });

    const num_indices = num_sides * 3;
    var indices: [num_indices]u32 = undefined;
    for (0..num_sides) |i| {
        indices[i * 3] = 0;
        indices[i * 3 + 1] = @as(u32, @intCast(i)) + 1;
        indices[i * 3 + 2] = @as(u32, @intCast(i)) + 2;
    }
    indices[num_indices - 1] = 1;
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_index_buffer).?, 0, u32, indices[0..]);
    return consumer_index_buffer;
}

pub fn createCircleVertexBuffer(
    gctx: *zgpu.GraphicsContext,
    comptime num_sides: u32,
    radius: f32,
) zgpu.BufferHandle {
    const buf = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = (num_sides + 1) * @sizeOf(f32) * 3,
    });
    var vertices: [num_sides + 1][3]f32 = undefined;
    const angle = 2 * std.math.pi / @as(f32, num_sides);
    vertices[0] = [3]f32{ 0, 0, 0 };
    for (0..num_sides) |i| {
        const current_angle = angle * @as(f32, @floatFromInt(i + 1));
        const x = @cos(current_angle) * radius;
        const y = @sin(current_angle) * radius;
        vertices[i + 1] = [3]f32{ x, y, 0 };
    }
    const resource = gctx.lookupResource(buf).?;
    gctx.queue.writeBuffer(resource, 0, [3]f32, vertices[0..]);
    return buf;
}
