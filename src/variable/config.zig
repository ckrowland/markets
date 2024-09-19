const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Wgpu = @import("wgpu.zig");

pub const cpi = .{
    .vs = @embedFile("shaders/vertex/consumer.wgsl"),
    .fs = @embedFile("shaders/fragment/fragment.wgsl"),
    .inst_size = @sizeOf(Consumer),
    .inst_attrs = &[_]Wgpu.RenderPipelineInfo.Attribute{
        .{
            .offset = @offsetOf(Consumer, "position"),
            .type = [4]f32,
        },
        .{
            .offset = @offsetOf(Consumer, "color"),
            .type = [4]f32,
        },
        .{
            .offset = @offsetOf(Consumer, "inventory"),
            .type = u32,
        },
    },
};

pub const ppi = .{
    .vs = @embedFile("shaders/vertex/producer.wgsl"),
    .fs = @embedFile("shaders/fragment/fragment.wgsl"),
    .inst_size = @sizeOf(Producer),
    .inst_attrs = &[_]Wgpu.RenderPipelineInfo.Attribute{
        .{
            .offset = @offsetOf(Producer.Params, "home"),
            .type = [4]f32,
        },
        .{
            .offset = @offsetOf(Producer.Params, "color"),
            .type = [4]f32,
        },
        .{
            .offset = @offsetOf(Producer, "inventory"),
            .type = u32,
        },
        .{
            .offset = @offsetOf(Producer.Params, "max_inventory"),
            .type = u32,
        },
    },
};

const common = @embedFile("shaders/compute/common.wgsl");
pub const ccpi = .{
    .cs = common ++ @embedFile("shaders/compute/consumer.wgsl"),
    .entry_point = "main",
};
pub const pcpi = .{
    .cs = common ++ @embedFile("shaders/compute/producer.wgsl"),
    .entry_point = "main",
};
