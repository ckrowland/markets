const std = @import("std");
const zgpu = @import("zgpu");
const zm = @import("zmath");
const Gctx = zgpu.GraphicsContext;
const wgpu = zgpu.wgpu;
const main = @import("resources.zig");
const Vertex = main.Vertex;
const GPUStats = main.GPUStats;
const DemoState = main.DemoState;
const Simulation = @import("simulation.zig");
const Consumer = Simulation.Consumer;
const CoordinateSize = Simulation.CoordinateSize;

pub const PipelineInfo = struct {
    pub const Attribute = struct {
        name: []const u8,
        type: type,
    };

    vs: [:0]const u8,
    fs: [:0]const u8,
    inst_type: type,
    inst_attrs: []const Attribute,
};

pub fn createUniformBindGroupLayout(gctx: *Gctx) zgpu.BindGroupLayoutHandle {
    return gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
    });
}

pub fn createUniformBindGroup(gctx: *Gctx, bgl: zgpu.BindGroupLayoutHandle) zgpu.BindGroupHandle {
    return gctx.createBindGroup(bgl, &.{
        .{ .binding = 0, .buffer_handle = gctx.uniforms.buffer, .offset = 0, .size = @sizeOf(zm.Mat) },
    });
}

fn getWgpuType(comptime T: type) !wgpu.VertexFormat {
    return switch (T) {
        f32 => .float32,
        [4]f32 => .float32x4,
        else => error.NoValidWgpuType, 
    };
}

pub fn createPipeline(
    gctx: *zgpu.GraphicsContext,
    layout: zgpu.PipelineLayoutHandle,
    comptime args: PipelineInfo,
) zgpu.RenderPipelineHandle {
    const vs_module = zgpu.createWgslShaderModule(gctx.device, args.vs, "vs");
    defer vs_module.release();

    const fs_module = zgpu.createWgslShaderModule(gctx.device, args.fs, "fs");
    defer fs_module.release();

    const color_targets = [_]wgpu.ColorTargetState{.{
        .format = zgpu.GraphicsContext.swapchain_format,
        .blend = &.{ .color = .{}, .alpha = .{} },
    }};

    const vertex_attributes = [_]wgpu.VertexAttribute{
        .{ .format = .float32x3, .offset = @offsetOf(Vertex, "position"), .shader_location = 0 },
    };

    const instance_attributes = init: {
        var arr: [args.inst_attrs.len]wgpu.VertexAttribute = undefined;
        inline for (args.inst_attrs) |attr, i| {
            arr[i] = .{
                .format = getWgpuType(attr.type) catch unreachable,
                .offset = @offsetOf(args.inst_type, attr.name),
                .shader_location = i + 1,
            }; 
        }
        break :init arr;
    };

    const vertex_buffers = [_]wgpu.VertexBufferLayout{
        .{
            .array_stride = @sizeOf(Vertex),
            .attribute_count = vertex_attributes.len,
            .attributes = &vertex_attributes,
            .step_mode = .vertex,
        },
        .{
            .array_stride = @sizeOf(args.inst_type),
            .attribute_count = instance_attributes.len,
            .attributes = &instance_attributes,
            .step_mode = .instance,
        },
    };

    const pipeline_descriptor = wgpu.RenderPipelineDescriptor{
        .vertex = wgpu.VertexState{
            .module = vs_module,
            .entry_point = "main",
            .buffer_count = vertex_buffers.len,
            .buffers = &vertex_buffers,
        },
        .primitive = wgpu.PrimitiveState{
            .front_face = .ccw,
            .cull_mode = .none,
            .topology = .triangle_list,
        },
        .depth_stencil = &wgpu.DepthStencilState{
            .format = .depth32_float,
            .depth_write_enabled = true,
            .depth_compare = .less,
        },
        .fragment = &wgpu.FragmentState{
            .module = fs_module,
            .entry_point = "main",
            .target_count = color_targets.len,
            .targets = &color_targets,
        },
    };

    return gctx.createRenderPipeline(layout, pipeline_descriptor);
}

pub fn createConsumerComputePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.ComputePipelineHandle {
    const common_cs = @embedFile("shaders/compute/common.wgsl");
    const cs = common_cs ++ @embedFile("shaders/compute/consumers.wgsl");
    const cs_module = zgpu.createWgslShaderModule(gctx.device, cs, "cs");
    defer cs_module.release();

    const pipeline_descriptor = wgpu.ComputePipelineDescriptor{
        .compute = wgpu.ProgrammableStageDescriptor{
            .module = cs_module,
            .entry_point = "main",
        },
    };

    return gctx.createComputePipeline(pipeline_layout, pipeline_descriptor);
}

pub fn createBindGroup(gctx: *zgpu.GraphicsContext, sim: DemoState.Simulation, compute_bgl: zgpu.BindGroupLayoutHandle, consumer_buffer: zgpu.BufferHandle, stats_buffer: zgpu.BufferHandle, size_buffer: zgpu.BufferHandle, splines_point_buffer: zgpu.BufferHandle, splines_buffer: zgpu.BufferHandle) zgpu.BindGroupHandle {
    const spb_info = gctx.lookupResourceInfo(splines_point_buffer).?;
    const sb_info = gctx.lookupResourceInfo(splines_buffer).?;
    return gctx.createBindGroup(compute_bgl, &[_]zgpu.BindGroupEntryInfo{
        .{
            .binding = 0,
            .buffer_handle = consumer_buffer,
            .offset = 0,
            .size = sim.consumers.items.len * @sizeOf(Consumer),
        },
        .{
            .binding = 1,
            .buffer_handle = stats_buffer,
            .offset = 0,
            .size = @sizeOf(GPUStats),
        },
        .{
            .binding = 2,
            .buffer_handle = size_buffer,
            .offset = 0,
            .size = @sizeOf(CoordinateSize),
        },
        .{
            .binding = 3,
            .buffer_handle = splines_point_buffer,
            .offset = 0,
            .size = spb_info.size,
        },
        .{
            .binding = 4,
            .buffer_handle = splines_buffer,
            .offset = 0,
            .size = sb_info.size,
        },
    });
}
