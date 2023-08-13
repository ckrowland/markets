const zgpu = @import("zgpu");

pub fn createVertexBuffer(
    gctx: *zgpu.GraphicsContext,
    width: f32,
) zgpu.BufferHandle {
    const vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = 8 * @sizeOf(f32) * 3,
    });

    const front_upper_left = [3]f32{ -width, width, width };
    const front_lower_left = [3]f32{ -width, -width, width };
    const front_upper_right = [3]f32{ width, width, width };
    const front_lower_right = [3]f32{ width, -width, width };

    const back_upper_left = [3]f32{ -width, width, -width };
    const back_lower_left = [3]f32{ -width, -width, -width };
    const back_upper_right = [3]f32{ width, width, -width };
    const back_lower_right = [3]f32{ width, -width, -width };

    const vertex_array = [8][3]f32{
        front_upper_left,
        front_lower_left,
        front_lower_right,
        front_upper_right,
        back_upper_left,
        back_lower_left,
        back_lower_right,
        back_upper_right,
    };

    gctx.queue.writeBuffer(
        gctx.lookupResource(vertex_buffer).?,
        0,
        [3]f32,
        vertex_array[0..],
    );
    return vertex_buffer;
}

pub fn createIndexBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const index = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = 6 * 6 * @sizeOf(u32),
    });

    var indices: [6 * 6]u32 = .{
        // front
        0, 1, 2, 2, 3, 0,
        // left - 0, 1, 5, 4
        0, 1, 5, 5, 4, 0,
        // right - 3, 2, 6, 7
        3, 2, 6, 6, 7, 3,
        // bottom - 1, 2, 6, 5
        1, 2, 6, 6, 5, 1,
        // top - 0, 4, 7, 3,
        0, 4, 7, 7, 3, 0,
        // back - 4, 5, 6, 7
        4, 5, 6, 6, 7, 4,
    };

    gctx.queue.writeBuffer(gctx.lookupResource(index).?, 0, u32, indices[0..]);
    return index;
}
