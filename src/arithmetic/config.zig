const Block = @import("block.zig");
const Wgpu = @import("../resources/wgpu.zig");

pub const block_render_pipeline = .{
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
};
