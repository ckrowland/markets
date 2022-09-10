const std = @import("std");
const math = std.math;
const random = std.crypto.random;
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const Vertex = @import("bloodstream.zig").Vertex;
const wgsl = @import("shaders.zig");

pub const Consumer = struct {
    position: @Vector(4, f32),
    color: @Vector(4, f32),
    velocity: @Vector(4, f32),
    consumption_rate: i32,
    inventory: i32,
    radius: f32,
    producer_id: i32,
};

pub fn createConsumers(self: *Simulation) void {
    var i: usize = 0;

    while (i < self.params.num_consumers) {
        const x = @intToFloat(f32, random.intRangeAtMost(i32, @floatToInt(i32, self.coordinate_size.min_x), @floatToInt(i32, self.coordinate_size.max_x)));
        const y = @intToFloat(f32, random.intRangeAtMost(i32, @floatToInt(i32, self.coordinate_size.min_y), @floatToInt(i32, self.coordinate_size.max_y)));
        const rand_x_vel = @intToFloat(f32, random.intRangeAtMost(i32, -5, 5));
        const rand_y_vel = @intToFloat(f32, random.intRangeAtMost(i32, -5, 5));
        const pos = @Vector(4, f32){ x, y, 0.0, 0.0 };
        const red = @Vector(4, f32){ 1.0, 0.0, 0.0, 0.0 };
        const vel = @Vector(4, f32){ rand_x_vel, rand_y_vel, 0.0, 0.0 };
        const c = Consumer{
            .position = pos,
            .color = red,
            .velocity = vel,
            .consumption_rate = self.params.consumption_rate,
            .inventory = 0,
            .radius = self.params.consumer_radius,
            .producer_id = 1000,
        };
        self.consumers.append(c) catch unreachable;
        i += 1;
    }
}

pub fn createConsumerIndexBuffer(gctx: *zgpu.GraphicsContext, comptime num_vertices: u32) zgpu.BufferHandle {
    const num_triangles = num_vertices - 1;
    const consumer_index_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = num_triangles * 3 * @sizeOf(u32),
    });
    const consumer_index_data = createConsumerIndexData(num_vertices - 1);
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_index_buffer).?, 0, i32, consumer_index_data[0..]);
    return consumer_index_buffer;
}

pub fn createConsumerVertexBuffer(gctx: *zgpu.GraphicsContext, radius: f32, comptime num_vertices: u32) zgpu.BufferHandle {
    const consumer_vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = num_vertices * @sizeOf(Vertex),
    });
    var consumer_vertex_data: [num_vertices]Vertex = undefined;
    const num_sides = @as(f32, num_vertices - 1);
    const angle = 2 * math.pi / num_sides;

    consumer_vertex_data[0] = .{ .position = [3]f32{ 0, 0, 0 }};
    var i: u32 = 1;
    while (i < num_vertices) {
        const current_angle = angle * @intToFloat(f32, i);
        const x = @cos(current_angle) * radius;
        const y = @sin(current_angle) * radius;
        consumer_vertex_data[i] = .{ .position = [3]f32{ x, y, 0 }};
        i += 1;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_vertex_buffer).?, 0, Vertex, consumer_vertex_data[0..]);
    return consumer_vertex_buffer;
}

pub fn createConsumerBuffer(gctx: *zgpu.GraphicsContext, consumers: array(Consumer)) zgpu.BufferHandle {
    const max_num_consumer = 10000;
    const consumer_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = max_num_consumer * @sizeOf(Consumer),
    });
    gctx.queue.writeBuffer(gctx.lookupResource(consumer_buffer).?, 0, Consumer, consumers.items[0..]);
    return consumer_buffer;
}

pub fn createConsumerIndexData(comptime num_triangles: u32) [num_triangles * 3]i32 {
    const vertices = num_triangles * 3;
    var consumer_index_data: [vertices]i32 = undefined;
    var i: usize = 0;
    while (i < num_triangles) {
        const idx = i * 3;
        const triangle_num = @intCast(i32, i);
        consumer_index_data[idx] = 0;
        consumer_index_data[idx + 1] = triangle_num + 1;
        consumer_index_data[idx + 2] = triangle_num + 2;
        i += 1;
    }
    consumer_index_data[vertices - 1] = 1;
    return consumer_index_data;
}

pub fn createConsumerPipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.RenderPipelineHandle {
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
        .{ .format = .float32x4, .offset = @offsetOf(Consumer, "position"), .shader_location = 1 },
        .{ .format = .float32x4, .offset = @offsetOf(Consumer, "color"), .shader_location = 2 },
    };

    const vertex_buffers = [_]wgpu.VertexBufferLayout{
        .{
            .array_stride = @sizeOf(Vertex),
            .attribute_count = vertex_attributes.len,
            .attributes = &vertex_attributes,
            .step_mode = .vertex,
        },
        .{
            .array_stride = @sizeOf(Consumer),
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

pub fn createConsumerComputePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.ComputePipelineHandle {
    const cs_module = zgpu.util.createWgslShaderModule(gctx.device, wgsl.cs, "cs");
    defer cs_module.release();

    const pipeline_descriptor = wgpu.ComputePipelineDescriptor{
        .compute = wgpu.ProgrammableStageDescriptor{
            .module = cs_module,
            .entry_point = "consumer_main",
        },
    };

    return gctx.createComputePipeline(pipeline_layout, pipeline_descriptor);
}
