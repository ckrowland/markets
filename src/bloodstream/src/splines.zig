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
const SplineCoords = @import("spline_coords.zig");
const SplinePointCoords = SplineCoords.SplinePointCoords;

pub const SplinePoint = struct {
    color: [4]f32,
    start_pos: [4]f32,
    current_pos: [4]f32,
    end_pos: [4]f32,
    step_size: [4]f32,
    radius: f32,
    to_start: u32,
    spline_id: u32,
    point_id: u32,
    len: u32,
    _padding: u64,
};

pub const Point = struct {
    color: [4]f32,
    position: [4]f32,
    radius: f32,
    _padding: u64,
};

pub fn createSplines(self: *Simulation) void {
    const outer_bloodstream = SplineCoords.outerBloodstream(self);
    const inner_bloodstream = SplineCoords.innerBloodstream(self);
    const top_left_heart = SplineCoords.topLeftHeart(self);
    const top_right_heart = SplineCoords.topRightHeart(self);
    const bottom_heart = SplineCoords.bottomHeart(self);

    var splines = array(array(SplinePointCoords)).init(self.allocator);
    splines.append(outer_bloodstream) catch unreachable;
    splines.append(inner_bloodstream) catch unreachable;
    splines.append(top_left_heart) catch unreachable;
    splines.append(top_right_heart) catch unreachable;
    splines.append(bottom_heart) catch unreachable;

    for (splines.items) |spline, idx| {
        for (spline.items) |p, i| {
            const len = @intCast(u32, spline.items.len);
            const color = [4]f32 { 0, 1, 1, 0 };
            const num_steps = 100;
            const radius = 20;

            const x = p.start[0];
            const y = p.start[1];
            const dx = p.delta[0];
            const dy = p.delta[1];

            const x_step_size = -dx / @intToFloat(f32, num_steps);
            const y_step_size = -dy / @intToFloat(f32, num_steps);
            const step_size = [4]f32{ x_step_size, y_step_size, 0, 0 };
            const start = [4]f32{ x, y, 0, 0 };
            const end = [4]f32{ x + dx, y + dy, 0, 0 };

            self.asplines.append(.{
                .color = color,
                .start_pos = start,
                .current_pos = start,
                .end_pos = end,
                .step_size = step_size,
                .radius = radius,
                .to_start = 0,
                .spline_id = @intCast(u32, idx),
                .point_id = @intCast(u32, i),
                .len = len,
                ._padding = 0,
            }) catch unreachable;
        }
    }

    outer_bloodstream.deinit();
    inner_bloodstream.deinit();
    top_left_heart.deinit();
    top_right_heart.deinit();
    bottom_heart.deinit();
    splines.deinit();
}


fn append1000BlankPoints(arr: *array(Point)) void {
    var t: u32 = 0;
    while (t < 1000) {
        arr.append(createBlankPoint()) catch unreachable;
        t += 1;
    }
}

fn createBlankPoint() Point {
    return Point{
        .position = [4]f32{ 0, 0, 0, 0 },
        .color = [4]f32{ 0, 0, 0, 0 },
        .radius = 0,
        ._padding = 0,
    };
}

pub fn getSplinePoint(i: f32, points: [4][2]f32) [2]f32 {
    const t = i;
    const tt = t * t;
    const ttt = tt * t;

    const q1 = -ttt + (2 * tt) - t;
    const q2 = (3 * ttt) - (5 * tt) + 2;
    const q3 = (-3 * ttt) + (4 * tt) + t;
    const q4 = ttt - tt;

    const p0x = points[0][0];
    const p1x = points[1][0];
    const p2x = points[2][0];
    const p3x = points[3][0];

    const p0y = points[0][1];
    const p1y = points[1][1];
    const p2y = points[2][1];
    const p3y = points[3][1];

    var tx = (p0x * q1) + (p1x * q2) + (p2x * q3) + (p3x * q4);
    var ty = (p0y * q1) + (p1y * q2) + (p2y * q3) + (p3y * q4);

    tx *= 0.5;
    ty *= 0.5;

    return [2]f32{ tx, ty };
}

pub fn createSplinesPointBuffer(gctx: *zgpu.GraphicsContext, splines: array(SplinePoint)) zgpu.BufferHandle {
    const splines_point_buffer = gctx.createBuffer(.{
        .usage = .{ 
            .copy_dst = true, 
            .copy_src = true, 
            .vertex = true,
            .storage = true,
        },
        .size = splines.items.len * @sizeOf(SplinePoint),
    });
    gctx.queue.writeBuffer(gctx.lookupResource(splines_point_buffer).?, 0, SplinePoint, splines.items[0..]);
    return splines_point_buffer;
}

pub fn createSplinesBuffer(gctx: *zgpu.GraphicsContext, points: array(SplinePoint), allocator: std.mem.Allocator) zgpu.BufferHandle {
    const num_points = points.items.len * 1000;
    const splines_point_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = num_points * @sizeOf(Point),
    });

    var point_data = array(Point).init(allocator);
    var t: f32 = 0;
    for (points.items) |sp, idx| {
        const first_point = sp.point_id == 0;
        const second_point = sp.point_id == 1;
        const last_point = sp.point_id == sp.len - 1;
        if (first_point or second_point or last_point) {
            append1000BlankPoints(&point_data);
            continue;
        }

        const spline_points = [4][2]f32{
            points.items[idx - 2].current_pos[0..2].*,
            points.items[idx - 1].current_pos[0..2].*,
            points.items[idx + 0].current_pos[0..2].*,
            points.items[idx + 1].current_pos[0..2].*,
        };
        while (t < 1000) {
            const position = getSplinePoint(t / 1000, spline_points);
            point_data.append(.{
                .color = [4]f32{ 0, 1, 1, 0 },
                .position = position ++ [_]f32{ 0, 0 },
                .radius = sp.radius - 15,
                ._padding = 0,
            }) catch unreachable;
            t += 1;
        }
        t = 0;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(splines_point_buffer).?, 0, Point, point_data.items[0..]);
    point_data.deinit();
    return splines_point_buffer;
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
            .offset = @offsetOf(Point, "position"),
            .shader_location = 1
        },
        .{
            .format = .float32,
            .offset = @offsetOf(Point, "radius"),
            .shader_location = 2
        },
        .{
            .format = .float32x4,
            .offset = @offsetOf(Point, "color"),
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
            .array_stride = @sizeOf(Point),
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
pub fn createSplinePointPipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.RenderPipelineHandle {
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
            .offset = @offsetOf(SplinePoint, "current_pos"),
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
