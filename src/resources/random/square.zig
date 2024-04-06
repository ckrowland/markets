const zgpu = @import("zgpu");

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
