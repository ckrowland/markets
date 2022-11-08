const std = @import("std");
const zgpu = @import("zgpu");
const zm = @import("zmath");
const Gctx = zgpu.GraphicsContext;
const wgpu = zgpu.wgpu;
const main = @import("resources.zig");
const GPUStats = main.GPUStats;
const DemoState = main.DemoState;
const Statistics = @import("statistics.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");

pub const RenderPipelineInfo = struct {
    pub const Attribute = struct {
        name: []const u8,
        type: type,
    };

    vs: [:0]const u8,
    fs: [:0]const u8,
    inst_type: type,
    inst_attrs: []const Attribute,
};

pub const ComputePipelineInfo = struct {
    cs: [:0]const u8,
    entry_point: [:0]const u8,
};


// Blank Buffers
pub fn createBuffer(
    gctx: *Gctx,
    comptime T: type,
    num: u32
) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{
            .copy_dst = true,
            .copy_src = true,
            .vertex = true,
            .storage = true
        },
        .size = num * @sizeOf(T),
    });
}

pub fn createMappedBuffer(
    gctx: *Gctx,
    comptime T: type,
    num: u32
) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = num * @sizeOf(T),
    });
}

// Bind Group Layouts
pub fn createUniformBindGroupLayout(gctx: *Gctx) zgpu.BindGroupLayoutHandle {
    return gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
    });
}

pub fn createComputeBindGroupLayout(gctx: *Gctx) zgpu.BindGroupLayoutHandle {
    return gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(1, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(2, .{ .compute = true }, .storage, false, 0),
    });
}


// Bind Groups
pub fn createUniformBindGroup(gctx: *Gctx) zgpu.BindGroupHandle {
    const bind_group_layout = createUniformBindGroupLayout(gctx);
    defer gctx.releaseResource(bind_group_layout);

    return gctx.createBindGroup(bind_group_layout, &.{
        .{
            .binding = 0,
            .buffer_handle = gctx.uniforms.buffer,
            .offset = 0,
            .size = @sizeOf(zm.Mat)
        },
    });
}

pub fn createComputeBindGroup(
    gctx: *zgpu.GraphicsContext,
    consumer_buffer: zgpu.BufferHandle,
    producer_buffer: zgpu.BufferHandle,
    stats_buffer: zgpu.BufferHandle
) zgpu.BindGroupHandle {

    const compute_bgl = createComputeBindGroupLayout(gctx);
    defer gctx.releaseResource(compute_bgl);

    const c_info = gctx.lookupResourceInfo(consumer_buffer) orelse unreachable;
    const p_info = gctx.lookupResourceInfo(producer_buffer) orelse unreachable;
    const s_info = gctx.lookupResourceInfo(stats_buffer) orelse unreachable;

    return gctx.createBindGroup(compute_bgl, &[_]zgpu.BindGroupEntryInfo{
        .{
            .binding = 0,
            .buffer_handle = consumer_buffer,
            .offset = 0,
            .size = c_info.size,
        },
        .{
            .binding = 1,
            .buffer_handle = producer_buffer,
            .offset = 0,
            .size = p_info.size,
        },
        .{
            .binding = 2,
            .buffer_handle = stats_buffer,
            .offset = 0,
            .size = s_info.size,
        },
    });
}

fn getWgpuType(comptime T: type) !wgpu.VertexFormat {
    return switch (T) {
        u32 => .uint32,
        f32 => .float32,
        [4]f32 => .float32x4,
        else => error.NoValidWgpuType, 
    };
}

pub fn createRenderPipeline(
    gctx: *zgpu.GraphicsContext,
    comptime args: RenderPipelineInfo,
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
        .{ .format = .float32x3, .offset = 0, .shader_location = 0 },
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
            .array_stride = @sizeOf(f32) * 3,
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
            .depth_compare = .less_equal,
        },
        .fragment = &wgpu.FragmentState{
            .module = fs_module,
            .entry_point = "main",
            .target_count = color_targets.len,
            .targets = &color_targets,
        },
    };

    const bind_group_layout = createUniformBindGroupLayout(gctx);
    defer gctx.releaseResource(bind_group_layout);

    const pipeline_layout = gctx.createPipelineLayout(&.{bind_group_layout});

    return gctx.createRenderPipeline(pipeline_layout, pipeline_descriptor);
}

pub fn createComputePipeline(gctx: *zgpu.GraphicsContext, cpi: ComputePipelineInfo) zgpu.ComputePipelineHandle {
    const compute_bgl = createComputeBindGroupLayout(gctx);
    defer gctx.releaseResource(compute_bgl);

    const compute_pl = gctx.createPipelineLayout(&.{compute_bgl});
    defer gctx.releaseResource(compute_pl);

    const cs_module = zgpu.createWgslShaderModule(gctx.device, cpi.cs, "cs");
    defer cs_module.release();

    const pipeline_descriptor = wgpu.ComputePipelineDescriptor{
        .compute = wgpu.ProgrammableStageDescriptor{
            .module = cs_module,
            .entry_point = cpi.entry_point,
        },
    };

    return gctx.createComputePipeline(compute_pl, pipeline_descriptor);
}

