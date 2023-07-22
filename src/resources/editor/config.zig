const Hover = @import("hover.zig");
const Consumer = @import("../consumer.zig");
const HoverConsumer = @import("consumer_hover.zig");
const Producer = @import("../producer.zig");
const Wgpu = @import("../wgpu.zig");

pub const cpi = .{
    .vs = @embedFile("../../shaders/vertex/consumer.wgsl"),
    .fs = @embedFile("../../shaders/fragment/fragment.wgsl"),
    .inst_type = Consumer,
    .inst_attrs = &[_]Wgpu.RenderPipelineInfo.Attribute{
        .{
            .name = "position",
            .type = [4]f32,
        },
        .{
            .name = "color",
            .type = [4]f32,
        },
        .{
            .name = "inventory",
            .type = u32,
        },
        .{
            .name = "demand_rate",
            .type = u32,
        },
    },
};

pub const ppi = .{
    .vs = @embedFile("../../shaders/vertex/producer.wgsl"),
    .fs = @embedFile("../../shaders/fragment/fragment.wgsl"),
    .inst_type = Producer,
    .inst_attrs = &[_]Wgpu.RenderPipelineInfo.Attribute{
        .{
            .name = "home",
            .type = [4]f32,
        },
        .{
            .name = "color",
            .type = [4]f32,
        },
        .{
            .name = "inventory",
            .type = u32,
        },
        .{
            .name = "max_inventory",
            .type = u32,
        },
    },
};
pub const hpi = .{
    .vs = @embedFile("../../shaders/vertex/hover.wgsl"),
    .fs = @embedFile("../../shaders/fragment/fragment.wgsl"),
    .inst_type = Hover,
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
pub const chpi = .{
    .vs = @embedFile("../../shaders/vertex/consumer_hover.wgsl"),
    .fs = @embedFile("../../shaders/fragment/fragment.wgsl"),
    .inst_type = HoverConsumer,
    .inst_attrs = &[_]Wgpu.RenderPipelineInfo.Attribute{
        .{
            .name = "home",
            .type = [4]f32,
        },
        .{
            .name = "color",
            .type = [4]f32,
        },
    },
};

const common = @embedFile("../../shaders/compute/common.wgsl");
pub const ccpi = .{
    .cs = common ++ @embedFile("../../shaders/compute/consumer.wgsl"),
    .entry_point = "main",
};
pub const pcpi = .{
    .cs = common ++ @embedFile("../../shaders/compute/producer.wgsl"),
    .entry_point = "main",
};
