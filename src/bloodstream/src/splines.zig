const std = @import("std");
const math = std.math;
const random = std.crypto.random;
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const Vertex = @import("bloodstream.zig").Vertex;
const wgsl = @import("shaders.zig");

pub const Spline = struct {
    points: []@Vector(2, f32),
    radius: f32,
};

pub const VertexColor = struct {
    position: [3]f32,
    color: [4]f32,
    radius: f32,
    radians: f32,
};

pub fn createSplines(self: *Simulation) void {
    const p1 = @Vector(2, f32){ 0, 0 };
    const p2 = @Vector(2, f32){ 200, 200 };
    const p3 = @Vector(2, f32){ 400, 200};
    const p4 = @Vector(2, f32){ 600, 0};
    const points = { p1, p2, p3, p4 };
    const radius = 5;
    const s = Spline{
        .points = points[0..],
        .radius = radius,
    };
    self.splines.append(s) catch unreachable;
}

pub fn createSplinesBuffer(gctx: *zgpu.GraphicsContext, splines: array(Spline)) zgpu.BufferHandle {
    const splines_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = splines.items.len * @sizeOf(Spline),
    });
    gctx.queue.writeBuffer(gctx.lookupResource(splines_buffer).?, 0, Spline, splines.items[0..]);
    return splines_buffer;
}

pub fn createSplinePointsBuffer(gctx: *zgpu.GraphicsContext, splines: array(Spline)) zgpu.BufferHandle {
    const max_num_squares = 1000;
    const spline_points_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = max_num_squares * @sizeOf(VertexColor),
    });
    var spline_vertex_data: [max_num_squares]VertexColor = undefined;
    for (spline.items[0].points) |p, i| {
        spline_vertex_data[i] = createVertexColor(p[0], p[1], 0, 0);
    }
    gctx.queue.writeBuffer(gctx.lookupResource(spline_points_buffer).?, 0, VertexColor, spline_vertex_data[0..]);
    return spline_points_buffer;
}

fn createVertexColor(x: f32, y: f32, radius: f32, radians: f32) VertexColor {
    return .{
        .position = [3]f32 {x, y, 0},
        .color = [4]f32{ 0, 1, 0, 0},
        .radius = radius,
        .radians = radians,
    };
}

pub fn createSplinePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.RenderPipelineHandle {
    const vs_module = zgpu.util.createWgslShaderModule(gctx.device, wgsl.vs, "vs");
    defer vs_module.release();

    const fs_module = zgpu.util.createWgslShaderModule(gctx.device, wgsl.fs, "fs");
    defer fs_module.release();

    const color_targets = [_]wgpu.ColorTargetState{.{
        .format = zgpu.GraphicsContext.swapchain_format,
        .blend = &.{ .color = .{}, .alpha = .{} },
    }};

    const vertex_attributes = [_]wgpu.VertexAttribute{
        .{ .format = .float32x3, .offset = @offsetOf(Vertex, "position"), .shader_location = 0 },
    };

    const instance_attributes = [_]wgpu.VertexAttribute{
        .{ .format = .float32x3, .offset = @offsetOf(VertexColor, "position"), .shader_location = 1 },
        .{ .format = .float32x4, .offset = @offsetOf(VertexColor, "color"), .shader_location = 2 },
    };

    const vertex_buffers = [_]wgpu.VertexBufferLayout{
        .{
            .array_stride = @sizeOf(Vertex),
            .attribute_count = vertex_attributes.len,
            .attributes = &vertex_attributes,
            .step_mode = .vertex,
        },
        .{
            .array_stride = @sizeOf(VertexColor),
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

    return gctx.createRenderPipeline(pipeline_layout, pipeline_descriptor);
}
