const Simulation = @import("simulation.zig");
const wgsl = @import("shaders.zig");
const Wgpu = @import("wgpu.zig");

pub const cpi = .{
    .vs = wgsl.vs,
    .fs = wgsl.fs,
    .inst_type = Simulation.Consumer,
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

pub const ppi = .{
    .vs = wgsl.vs,
    .fs = wgsl.fs,
    .inst_type = Simulation.Producer,
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
            .name = "max_inventory",
            .type = u32,
        },
    },
};

pub const ccpi = .{
    .cs = wgsl.cs,
    .entry_point = "consumer_main",
};

pub const pcpi = .{
    .cs = wgsl.cs,
    .entry_point = "producer_main",
};
