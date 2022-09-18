const std = @import("std");
const math = std.math;
const random = std.crypto.random;
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const Vertex = @import("bloodstream.zig").Vertex;
const wgsl = @import("shaders.zig");

pub const AnimatedSpline = struct {
    start: Spline,
    current: Spline,
    end: Spline,
    step_size: f32,
    to_start: bool,
    len: u32,
};

pub const Spline = struct {
    radius: f32,
    len: u32,
    points: [10][2]f32,
};

pub const VertexColor = struct {
    position: [3]f32,
    color: [4]f32,
    radius: f32,
    radians: f32,
};

pub fn updateHeartSpline(spline: AnimatedSpline) AnimatedSpline {
    var updated_spline = spline;
    var diff: f32 = 0;
    var step_size: f32 = 0;
    var i: u32 = 0;
    while (i < spline.len) {
        const p = spline.current.points[i];
        const e = spline.end.points[i];
        i += 1;
        if (p[0] == e[0] and p[1] == e[1] ) {
            continue;
        } else {
            break;
        }
        if (i == spline.len - 1) {
            updated_spline.to_start = true;
        }
    }

    i = 0;
    while (i < spline.len) {
        const p = spline.current.points[i];
        const s = spline.start.points[i];
        i += 1;
        if (p[0] == s[0] or p[1] == s[1] ) {
            continue;
        } else {
            break;
        }
        if (i == spline.len - 1) {
            updated_spline.to_end = true;
        }
    }

    i = 0;
    while (i < spline.len) {
        const start_point = spline.start.points[i];
        const current_point = spline.current.points[i];
        const end_point = spline.end.points[i];
        if (spline.to_start) {
            if (start_point[0] < current_point[0]) {
                diff = current_point[0] - start_point[0];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][0] -= step_size;
            } else if (start_point[0] > current_point[0]) {
                diff = start_point[0] - current_point[0];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][0] += step_size;
            }
            diff = 0;
            step_size = 0;
            if (start_point[1] < current_point[1]) {
                diff = current_point[1] - start_point[1];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][1] -= step_size;
            } else if (start_point[0] > current_point[0]) {
                diff = start_point[1] - current_point[1];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][1] += step_size;
            }
        } else {
            if (end_point[0] < current_point[0]) {
                diff = current_point[0] - end_point[0];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][0] -= step_size;
            } else if (end_point[0] > current_point[0]) {
                diff = end_point[0] - current_point[0];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][0] += step_size;
            }
            diff = 0;
            step_size = 0;
            if (end_point[1] < current_point[1]) {
                diff = current_point[1] - end_point[1];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][1] -= step_size;
            } else if (end_point[0] > current_point[0]) {
                diff = end_point[1] - current_point[1];
                step_size = @minimum(diff, spline.step_size);
                updated_spline.current.points[i][1] += step_size;
            }
        }
        i += 1;
    }
    return updated_spline;
}

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
    const outer = [10][2]f32{
        [2]f32{cx+offset, cy},
        [2]f32{cx+offset, cy+height},
        [2]f32{cx+width, cy+height},
        [2]f32{cx+width, cy},
        [2]f32{cx, cy-height},
        [2]f32{cx-width, cy},
        [2]f32{cx-width, cy+height},
        [2]f32{cx-offset, cy+height},
        [2]f32{cx-offset, cy},
        [2]f32{0, 0},
    };
    width = 200;
    height = 200;
    const inner = [10][2]f32{
        [2]f32{cx+offset, cy},
        [2]f32{cx+offset, cy+height},
        [2]f32{cx+width, cy+height},
        [2]f32{cx+width, cy},
        [2]f32{cx, cy-height},
        [2]f32{cx-width, cy},
        [2]f32{cx-width, cy+height},
        [2]f32{cx-offset, cy+height},
        [2]f32{cx-offset, cy},
        [2]f32{0, 0},
    };
    const radius = 5;
    const len = 9;
    const start = Spline {
        .points = outer,
        .len = len,
        .radius = radius
    };
    const end = Spline {
        .points = inner,
        .len = len,
        .radius = radius
    };
    self.splines.animated.append(.{
        .start = start,
        .current = start,
        .end = end,
        .step_size = 1,
        .to_start = false,
        .len = len,
    }) catch unreachable;
    self.splines.stationary.append(start) catch unreachable;
}

fn createSplineBloodstream(self: *Simulation) void {
    const c = self.coordinate_size;
    const len_x = @fabs(c.min_x) + @fabs(c.max_x);
    const len_y = @fabs(c.min_y) + @fabs(c.max_y);
    const cx = c.min_x + (len_x / 2);
    const cy = c.min_y + (len_y / 2);
    var offset: f32 = 100;
    var width: f32 = 700 + offset;
    var height: f32 = 700;
    const points = [10][2]f32{
        [2]f32{cx+offset, cy},
        [2]f32{cx+offset, cy+height},
        [2]f32{cx+width, cy+height},
        [2]f32{cx+width, cy},
        [2]f32{cx, cy-height},
        [2]f32{cx-width, cy},
        [2]f32{cx-width, cy+height},
        [2]f32{cx-offset, cy+height},
        [2]f32{cx-offset, cy},
        [2]f32{0, 0},
    };
    const radius = 5;
    const s = Spline{
        .points = points,
        .radius = radius,
        .len = 9,
    };
    self.splines.append(s) catch unreachable;

    offset = 200;
    width = 600;
    height = 600;
    const points2 = [10][2]f32{
        [2]f32{cx+offset, cy},
        [2]f32{cx+offset, cy+height},
        [2]f32{cx+width, cy+height},
        [2]f32{cx+width, cy},
        [2]f32{cx, cy-height},
        [2]f32{cx-width, cy},
        [2]f32{cx-width, cy+height},
        [2]f32{cx-offset, cy+height},
        [2]f32{cx-offset, cy},
        [2]f32{0, 0},
    };
    const s2 = Spline{
        .points = points2,
        .radius = radius,
        .len = 9,
    };
    self.splines.append(s2) catch unreachable;
}

pub fn getSplinePoint(i: f32, points: [10][2]f32) [2]f32 {
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

    const tx = 0.5 * (p[p0][0] * q1 + p[p1][0] * q2 + p[p2][0] * q3 + p[p3][0] * q4);
    const ty = 0.5 * (p[p0][1] * q1 + p[p1][1] * q2 + p[p2][1] * q3 + p[p3][1] * q4);

    return [2]f32{ tx, ty };
}

pub fn createAnimatedSplinesBuffer(gctx: *zgpu.GraphicsContext, splines: array(AnimatedSpline)) zgpu.BufferHandle {
    const splines_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = splines.items.len * @sizeOf(Spline),
    });
    var current_splines: [splines.items.len]Spline = undefined;
    for (splines.items) |as, i| {
        current_splines[i] = as.current;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(splines_buffer).?, 0, Spline, current_splines[0..]);
    return splines_buffer;
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
    var points_buffer: [max_num_squares]VertexColor = undefined;
    var buffer_idx: u32 = 0;
    var point_idx: u32 = 0;
    for (splines.items) |s| {
        while (point_idx < s.len) {
            const p = s.points[point_idx];
            points_buffer[buffer_idx] = createVertexColor(p[0], p[1], 20, 0);
            point_idx += 1;
            buffer_idx += 1;
        }
        point_idx = 0;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(spline_points_buffer).?, 0, VertexColor, points_buffer[0..]);
    return spline_points_buffer;
}

pub fn createSplinesSquaresBuffer(gctx: *zgpu.GraphicsContext, splines: array(Spline)) zgpu.BufferHandle {
    const max_num_squares = 100000;
    const splines_square_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = max_num_squares * @sizeOf(VertexColor),
    });
    var spline_vertex_data: [max_num_squares]VertexColor = undefined;
    var t: f32 = 0;
    var i: u32 = 0;
    for (splines.items) |s| {
        const num_points = @intToFloat(f32, s.len - 3);
        while (t < num_points) {
            const p = getSplinePoint(t, s.points);
            spline_vertex_data[i] = createVertexColor(p[0], p[1], s.radius, 0);
            i += 1;
            t += 0.001;
        }
        t = 0;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(splines_square_buffer).?, 0, VertexColor, spline_vertex_data[0..]);
    return splines_square_buffer;
}
fn createVertexColor(x: f32, y: f32, radius: f32, radians: f32) VertexColor {
    return .{
        .position = [3]f32 {x, y, 0},
        .color = [4]f32{ 0, 1, 1, 0},
        .radius = radius,
        .radians = radians,
    };
}

pub fn createSplinePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.RenderPipelineHandle {
    const vs_module = zgpu.util.createWgslShaderModule(gctx.device, wgsl.line_vs, "vs");
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
        .{ .format = .float32, .offset = @offsetOf(VertexColor, "radius"), .shader_location = 3 },
        .{ .format = .float32, .offset = @offsetOf(VertexColor, "radians"), .shader_location = 4 },
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
