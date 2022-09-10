const std = @import("std");
const math = std.math;
const random = std.crypto.random;
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const Vertex = @import("bloodstream.zig").Vertex;
const wgsl = @import("shaders.zig");

pub const Line = struct {
    color: @Vector(4, f32),
    start: @Vector(2, f32),
    end: @Vector(2, f32),
    radius: f32,
    num_squares: u32,
};

pub const VertexColor = struct {
    position: [3]f32,
    color: [4]f32,
    radius: f32,
    radians: f32,
};

pub fn createLines(self: *Simulation) void {
    const start = @Vector(2, f32){ -200, -200 };
    const end = @Vector(2, f32){ 800, 800};
    const green = @Vector(4, f32){ 0.0, 1.0, 0.0, 0.0 };
    const radius = 5;
    var dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const len = @sqrt((dx * dx) + (dy * dy));
    const num_squares = @ceil(len / (radius * 2));
    const l = Line{
        .color = green,
        .start = start,
        .end = end,
        .radius = radius,
        .num_squares = @floatToInt(u32, num_squares),
    };
    self.lines.append(l) catch unreachable;
}

pub fn createLinesBuffer(gctx: *zgpu.GraphicsContext, lines: array(Line)) zgpu.BufferHandle {
    const lines_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = lines.items.len * @sizeOf(Line),
    });
    gctx.queue.writeBuffer(gctx.lookupResource(lines_buffer).?, 0, Line, lines.items[0..]);
    return lines_buffer;
}

pub fn createSquareVertexBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const square_vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = 6 * @sizeOf(Vertex),
    });

    const width = 1;
    const upper_left = [3]f32{ -width, width, 0.0 };
    const lower_left = [3]f32{ -width, -width, 0.0 };
    const upper_right = [3]f32{ width, width, 0.0 };
    const lower_right = [3]f32{ width, -width, 0.0 };

    const square_vertex_array = [6]Vertex
        { .{ .position = upper_left, },
          .{ .position = lower_left, },
          .{ .position = lower_right, },
          .{ .position = lower_right, },
          .{ .position = upper_right, },
          .{ .position = upper_left, }, };

    gctx.queue.writeBuffer(gctx.lookupResource(square_vertex_buffer).?, 0, Vertex, square_vertex_array[0..]);
    return square_vertex_buffer;
}

pub fn createSquarePositionBuffer(gctx: *zgpu.GraphicsContext, lines: array(Line)) zgpu.BufferHandle {
    const max_num_squares = 1000;
    const square_position_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = max_num_squares * @sizeOf(VertexColor),
    });
    const line = lines.items[0];
    const dx = line.end[0] - line.start[0];
    const dy = line.end[1] - line.start[1];
    var radians: f32 = 0;
    if (dx == 0) {
        if (dy > 0) {
            radians = math.pi / 2.0;
        } else if (dy < 0) {
            radians = (3 * math.pi) / 2.0;
        } else {
            radians = 0;
        }
    } else {
        radians = math.atan(dy / dx);
    }
    var line_vertex_data: [max_num_squares]VertexColor = undefined;
    var i: u32 = 0;
    const f_num_squares = @intToFloat(f32, line.num_squares);
    const dx_square = (line.end[0] - line.start[0]) / f_num_squares;
    const dy_square = (line.end[1] - line.start[1]) / f_num_squares;
    var sx = line.start[0] + (dx_square / 2);
    var sy = line.start[1] + (dy_square / 2);
    while (i < line.num_squares) {
        line_vertex_data[i] = createVertexColor(sx, sy, line.radius, radians);
        sx += dx_square;
        sy += dy_square;
        i += 1;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(square_position_buffer).?, 0, VertexColor, line_vertex_data[0..]);
    return square_position_buffer;
}

fn createVertexColor(x: f32, y: f32, radius: f32, radians: f32) VertexColor {
    return .{
        .position = [3]f32 {x, y, 0},
        .color = [4]f32{ 0, 1, 0, 0},
        .radius = radius,
        .radians = radians,
    };
}

pub fn createLinePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.RenderPipelineHandle {
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
