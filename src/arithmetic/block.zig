const zgpu = @import("zgpu");
const Gctx = zgpu.GraphicsContext;
const Wgpu = @import("../resources/wgpu.zig");

const Block = struct {
    position: [4]f32,
    color: [4]f32,
};

pub const GraphicsObjects = struct {
    block: Wgpu.GraphicsObject,
    border: Wgpu.GraphicsObject,
};

pub fn init(gctx: *zgpu.GraphicsContext) GraphicsObjects {
    return .{
        .block = .{
            .render_pipeline = Wgpu.createRenderPipeline(
                gctx,
                .{
                    .vs = @embedFile("shaders/vertex/block.wgsl"),
                    .fs = @embedFile("shaders/fragment/basic.wgsl"),
                    .inst_type = Block,
                    .inst_attrs = &[_]Wgpu.RenderPipelineInfo.Attribute{
                        .{
                            .name = "position",
                            .type = [4]f32,
                        },
                        .{
                            .name = "color",
                            .type = [4]f32,
                        },
                    },
                },
            ),
            .attribute_buffer = createOneBlockBuffer(gctx),
            .vertex_buffer = createVertexBuffer(gctx, 30),
            .index_buffer = createIndexBuffer(gctx),
            .size_of_struct = @sizeOf(Block),
        },
        .border = .{
            .render_pipeline = Wgpu.createRenderPipeline(
                gctx,
                .{
                    .vs = @embedFile("shaders/vertex/block.wgsl"),
                    .fs = @embedFile("shaders/fragment/basic.wgsl"),
                    .inst_type = Block,
                    .inst_attrs = &[_]Wgpu.RenderPipelineInfo.Attribute{
                        .{
                            .name = "position",
                            .type = [4]f32,
                        },
                        .{
                            .name = "color",
                            .type = [4]f32,
                        },
                    },
                    .primitive_topology = .line_strip,
                },
            ),
            .attribute_buffer = createOneBlockBuffer(gctx),
            .vertex_buffer = createVertexBuffer(gctx, 30),
            .index_buffer = createBorderIndexBuffer(gctx),
            .size_of_struct = @sizeOf(Block),
        },
    };
}

pub fn createOneBlockBuffer(gctx: *Gctx) zgpu.BufferHandle {
    const block_buffer = Wgpu.createBuffer(gctx, Block, 10);
    var blocks: [1]Block = .{
        Block{
            .position = .{ 0, 0, 0, 0 },
            .color = .{ 1, 0, 0, 0 },
        },
    };
    gctx.queue.writeBuffer(
        gctx.lookupResource(block_buffer).?,
        0,
        Block,
        blocks[0..],
    );
    return block_buffer;
}

pub fn createVertexBuffer(
    gctx: *zgpu.GraphicsContext,
    width: f32,
) zgpu.BufferHandle {
    const vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = 8 * @sizeOf(f32) * 3,
    });

    const front_upper_left = [3]f32{ -width, width, -width };
    const front_lower_left = [3]f32{ -width, -width, -width };
    const front_upper_right = [3]f32{ width, width, -width };
    const front_lower_right = [3]f32{ width, -width, -width };

    const back_upper_left = [3]f32{ -width, width, width };
    const back_lower_left = [3]f32{ -width, -width, width };
    const back_upper_right = [3]f32{ width, width, width };
    const back_lower_right = [3]f32{ width, -width, width };

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

pub fn createBorderIndexBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const index = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = 10 * @sizeOf(u32),
    });

    var indices: [10]u32 = .{
        // front
        0, 1, 2, 3, 0, 4, 5, 6, 7, 3,
    };

    gctx.queue.writeBuffer(gctx.lookupResource(index).?, 0, u32, indices[0..]);
    return index;
}
