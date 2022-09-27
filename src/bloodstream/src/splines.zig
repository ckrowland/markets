const std = @import("std");
const math = std.math;
const random = std.crypto.random;
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const main = @import("bloodstream.zig");
const Vertex = main.Vertex;
const DemoState = main.DemoState;
const wgsl = @import("shaders.zig");

pub const AnimatedSpline = struct {
    current: [10]SplinePoint,
    start: [10]SplinePoint,
    end: [10]SplinePoint,
    len: u32,
    to_start: u32,
    _padding: u136,
};

pub const Spline = struct {
    points: [10]SplinePoint,
    len: u32,
    _padding: u168,
};

pub const SplinePoint = struct {
    color: [4]f32,
    position: [4]f32,
    radius: f32,
    step_size: f32,
    _padding: u64,
};

pub fn createSplines(self: *Simulation) void {
    createSplineHeart(self);
    //createSplineBloodstream(self);
}

fn createSplineHeart(self: *Simulation) void {
    const c = self.coordinate_size;
    const len_x = @fabs(c.min_x) + @fabs(c.max_x);
    const len_y = @fabs(c.min_y) + @fabs(c.max_y);
    const cx = c.min_x + (len_x / 2);
    const cy = c.min_y + (len_y / 2);
    var offset: f32 = 30;
    var width: f32 = 300;
    var height: f32 = 300;
    const radius = 20;
    const outer = [10]SplinePoint{
        createSplinePoint(cx+offset, cy, radius),
        createSplinePoint(cx+offset, cy+height, radius),
        createSplinePoint(cx+width, cy+height, radius),
        createSplinePoint(cx+width, cy, radius),
        createSplinePoint(cx, cy-height, radius),
        createSplinePoint(cx-width, cy, radius),
        createSplinePoint(cx-width, cy+height, radius),
        createSplinePoint(cx-offset, cy+height, radius),
        createSplinePoint(cx-offset, cy, radius),
        createSplinePoint(0, 0, radius),
    };

    width = 200;
    height = 200;
    const inner = [10]SplinePoint{
        createSplinePoint(cx+offset, cy, radius),
        createSplinePoint(cx+offset, cy+height, radius),
        createSplinePoint(cx+width, cy+height, radius),
        createSplinePoint(cx+width, cy, radius),
        createSplinePoint(cx, cy-height, radius),
        createSplinePoint(cx-width, cy, radius),
        createSplinePoint(cx-width, cy+height, radius),
        createSplinePoint(cx-offset, cy+height, radius),
        createSplinePoint(cx-offset, cy, radius),
        createSplinePoint(0, 0, radius),
    };

    const len = 9;
    self.asplines.append(.{
        .start = outer,
        .current = outer,
        .end = inner,
        .len = len,
        .to_start = 0,
        ._padding = 0,
    }) catch unreachable;
}

fn createSplinePoint(x: f32, y: f32, radius: f32) SplinePoint {
    return SplinePoint{
        .radius = radius,
        .color = [4]f32{ 0, 1, 1, 0 },
        .position = [4]f32{ x, y, 0, 0 },
        .step_size = 1,
        ._padding = 0,
    };
}

fn createSplineBloodstream(self: *Simulation) void {
    const c = self.coordinate_size;
    const len_x = @fabs(c.min_x) + @fabs(c.max_x);
    const len_y = @fabs(c.min_y) + @fabs(c.max_y);
    const cx = c.min_x + (len_x / 2);
    const cy = c.min_y + (len_y / 2);
    const radius = 20;
    var offset: f32 = 100;
    var width: f32 = 700 + offset;
    var height: f32 = 700;
    const outer = [10]SplinePoint{
        createSplinePoint(cx+offset, cy, radius),
        createSplinePoint(cx+offset, cy+300, radius),
        createSplinePoint(cx+width, cy+height, radius),
        createSplinePoint(cx+width, cy, radius),
        createSplinePoint(cx, cy-height, radius),
        createSplinePoint(cx-width, cy, radius),
        createSplinePoint(cx-width, cy+height, radius),
        createSplinePoint(cx-offset, cy+300, radius),
        createSplinePoint(cx-offset, cy, radius),
        createSplinePoint(0, 0, radius),
    };
    const outer_end = [10]SplinePoint{
        createSplinePoint(cx+offset, cy, radius),
        createSplinePoint(cx+offset, cy+200, radius),
        createSplinePoint(cx+width, cy+height, radius),
        createSplinePoint(cx+width, cy, radius),
        createSplinePoint(cx, cy-height, radius),
        createSplinePoint(cx-width, cy, radius),
        createSplinePoint(cx-width, cy+height, radius),
        createSplinePoint(cx-offset, cy+200, radius),
        createSplinePoint(cx-offset, cy, radius),
        createSplinePoint(0, 0, radius),
    };
    const outer_as = AnimatedSpline{
        .start = outer,
        .current = outer,
        .end = outer_end,
        .len = 9,
        .to_start = 0,
        ._padding = 0,
    };
    self.asplines.append(outer_as) catch unreachable;

    //offset = 200;
    //width = 600;
    //height = 600;
    //const inner = [10]SplinePoint{
    //    createSplinePoint(cx+offset, cy, radius),
    //    createSplinePoint(cx+offset, cy+300, radius),
    //    createSplinePoint(cx+width, cy+height, radius),
    //    createSplinePoint(cx+width, cy, radius),
    //    createSplinePoint(cx, cy-height, radius),
    //    createSplinePoint(cx-width, cy, radius),
    //    createSplinePoint(cx-width, cy+height, radius),
    //    createSplinePoint(cx-offset, cy+300, radius),
    //    createSplinePoint(cx-offset, cy, radius),
    //    createSplinePoint(0, 0, radius),
    //};
    //const inner_end = [10]SplinePoint{
    //    createSplinePoint(cx+offset, cy, radius),
    //    createSplinePoint(cx+offset, cy+200, radius),
    //    createSplinePoint(cx+width, cy+height, radius),
    //    createSplinePoint(cx+width, cy, radius),
    //    createSplinePoint(cx, cy-height, radius),
    //    createSplinePoint(cx-width, cy, radius),
    //    createSplinePoint(cx-width, cy+height, radius),
    //    createSplinePoint(cx-offset, cy+200, radius),
    //    createSplinePoint(cx-offset, cy, radius),
    //    createSplinePoint(0, 0, radius),
    //};
    //const inner_as = AnimatedSpline{
    //    .start = inner,
    //    .current = inner,
    //    .end = inner_end,
    //    .len = 9,
    //    .to_start = 0,
    //    ._padding = 0,
    //};
    //self.asplines.append(inner_as) catch unreachable;
}

pub fn getSplinePoint(i: f32, points: [10]SplinePoint) [4]f32 {
    const p0: u32 = @floatToInt(u32, i);
    const p1 = p0 + 1;
    const p2 = p1 + 1;
    const p3 = p2 + 1;

    const t = i - @floor(i);
    const tt = t * t;
    const ttt = tt * t;

    const q1 = -ttt + (2 * tt) - t;
    const q2 = (3 * ttt) - (5 * tt) + 2;
    const q3 = (-3 * ttt) + (4 * tt) + t;
    const q4 = ttt - tt;

    const p = points;

    const tx = 0.5 * (p[p0].position[0] * q1 + p[p1].position[0] * q2 + p[p2].position[0] * q3 + p[p3].position[0] * q4);
    const ty = 0.5 * (p[p0].position[1] * q1 + p[p1].position[1] * q2 + p[p2].position[1] * q3 + p[p3].position[1] * q4);

    return [4]f32{ tx, ty, 0, 0 };
}

pub fn createAnimatedSplinesBuffer(gctx: *zgpu.GraphicsContext, splines: array(AnimatedSpline)) zgpu.BufferHandle {
    const animated_splines_buffer = gctx.createBuffer(.{
        .usage = .{ 
            .copy_dst = true, 
            .copy_src = true, 
            .vertex = true,
            .storage = true,
        },
        .size = splines.items.len * @sizeOf(AnimatedSpline),
    });
    gctx.queue.writeBuffer(gctx.lookupResource(animated_splines_buffer).?, 0, AnimatedSpline, splines.items[0..]);
    return animated_splines_buffer;
}

pub fn createSplinePointsBuffer(gctx: *zgpu.GraphicsContext, splines: array(AnimatedSpline)) zgpu.BufferHandle {
    //const max_num_points = 1000;
    var num_points: u32 = 0;
    for (splines.items) |as| {
        num_points += as.len;
    }
    const spline_points_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = num_points * @sizeOf(SplinePoint),
    });
    var points_buffer: [1000]SplinePoint = undefined;
    var buffer_idx: u32 = 0;
    var point_idx: u32 = 0;
    for (splines.items) |as| {
        while (point_idx < as.len) {
            const p = as.current[point_idx];
            points_buffer[buffer_idx] = p;
            point_idx += 1;
            buffer_idx += 1;
        }
        point_idx = 0;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(spline_points_buffer).?, 0, SplinePoint, points_buffer[0..num_points]);
    return spline_points_buffer;
}

pub fn createSplinesBuffer(gctx: *zgpu.GraphicsContext, asplines: array(AnimatedSpline)) zgpu.BufferHandle {
    const max_num_curves = 7;
    const max_num_points = max_num_curves * 1000;
    const splines_point_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = asplines.items.len * max_num_points * @sizeOf(SplinePoint),
    });
    var spline_vertex_data: [max_num_points]SplinePoint = undefined;
    var t: f32 = 0;
    var i: u32 = 0;
    for (asplines.items) |as| {
        const num_curves = as.len - 3;
        const num_points = @intToFloat(f32, num_curves);
        while (t < num_points) {
            const p = getSplinePoint(t, as.current);
            const prev_point = @floatToInt(u32, t);
            const radius = as.current[prev_point].radius - 15;
            spline_vertex_data[i] = createSplinePoint(p[0], p[1], radius);
            i += 1;
            t += 0.001;
        }
        t = 0;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(splines_point_buffer).?, 0, SplinePoint, spline_vertex_data[0..]);
    return splines_point_buffer;
}

// STATIONARY SPLINES DRAWING BUFFERS
pub fn createStationarySplinesBuffer(gctx: *zgpu.GraphicsContext, splines: array(Spline)) zgpu.BufferHandle {
    const max_num_curves = 7;
    const max_num_splines = 2;
    const max_num_points = max_num_curves * 10000 * max_num_splines;
    const stationary_splines_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = max_num_points * @sizeOf(SplinePoint),
    });

    var spline_vertex_data: [max_num_points]SplinePoint = undefined;
    var t: f32 = 0;
    var i: u32 = 0;
    var point_idx: u32 = 0;
    for (splines.items) |s| {
        while (point_idx < s.len) {
            spline_vertex_data[i] = s.points[point_idx];
            point_idx += 1;
            i += 1;
        }
        point_idx = 0;

        const num_curves = s.len - 3;
        const num_points = @intToFloat(f32, num_curves);
        while (t < num_points) {
            const p = getSplinePoint(t, s.points);
            const prev_point = @floatToInt(u32, t);
            const radius = s.points[prev_point].radius - 15;
            spline_vertex_data[i] = createSplinePoint(p[0], p[1], radius);
            i += 1;
            t += 0.001;
        }
        t = 0;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(stationary_splines_buffer).?, 0, SplinePoint, spline_vertex_data[0..]);
    return stationary_splines_buffer;
}

pub fn createAnimatedSplineBindGroup(gctx: *zgpu.GraphicsContext, sim: Simulation, compute_bgl: zgpu.BindGroupLayoutHandle, animated_splines_buffer: zgpu.BufferHandle) zgpu.BindGroupHandle {
    return gctx.createBindGroup(compute_bgl, &[_]zgpu.BindGroupEntryInfo{
        .{
            .binding = 0,
            .buffer_handle = animated_splines_buffer,
            .offset = 0,
            .size = sim.splines.animated.items.len * @sizeOf(AnimatedSpline),
        },
    });
}

pub fn createSplinePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.RenderPipelineHandle {
    const vs_module = zgpu.util.createWgslShaderModule(gctx.device, wgsl.spline_vs, "spline_vs");
    defer vs_module.release();

    const fs_module = zgpu.util.createWgslShaderModule(gctx.device, wgsl.fs, "fs");
    defer fs_module.release();

    const color_targets = [_]wgpu.ColorTargetState{.{
        .format = zgpu.GraphicsContext.swapchain_format,
        .blend = &.{ .color = .{}, .alpha = .{} },
    }};

    const vertex_attributes = [_]wgpu.VertexAttribute{
        .{
            .format = .float32x3,
            .offset = @offsetOf(Vertex, "position"),
            .shader_location = 0
        },
    };

    const spline_instance_attributes = [_]wgpu.VertexAttribute{
        .{
            .format = .float32x4,
            .offset = @offsetOf(SplinePoint, "position"),
            .shader_location = 1
        },
        .{
            .format = .float32,
            .offset = @offsetOf(SplinePoint, "radius"),
            .shader_location = 2
        },
        .{
            .format = .float32x4,
            .offset = @offsetOf(SplinePoint, "color"),
            .shader_location = 3
        },
    };

    const vertex_buffers = [_]wgpu.VertexBufferLayout{
        .{
            .array_stride = @sizeOf(Vertex),
            .attribute_count = vertex_attributes.len,
            .attributes = &vertex_attributes,
            .step_mode = .vertex,
        },
        .{
            .array_stride = @sizeOf(SplinePoint),
            .attribute_count = spline_instance_attributes.len,
            .attributes = &spline_instance_attributes,
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

pub fn createAnimatedSplineComputePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.ComputePipelineHandle {
    const cs_module = zgpu.util.createWgslShaderModule(gctx.device, wgsl.cs, "cs");
    defer cs_module.release();

    const pipeline_descriptor = wgpu.ComputePipelineDescriptor{
        .compute = wgpu.ProgrammableStageDescriptor{
            .module = cs_module,
            .entry_point = "animated_spline_main",
        },
    };

    return gctx.createComputePipeline(pipeline_layout, pipeline_descriptor);
}
