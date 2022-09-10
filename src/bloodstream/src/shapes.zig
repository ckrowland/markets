const main = @import("bloodstream.zig");
const Vertex = main.Vertex;
const GPUStats = main.GPUStats;
const std = @import("std");
const math = std.math;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const Producer = Simulation.Producer;
const CoordinateSize = Simulation.CoordinateSize;
const Lines = @import("lines.zig");
const Line = Lines.Line;
const wgsl = @import("shaders.zig");
const array = std.ArrayList;
const Consumers = @import("consumers.zig");
const Consumer = Consumers.Consumer;

pub fn createCoordinateSizeBuffer(gctx: *zgpu.GraphicsContext, size: CoordinateSize) zgpu.BufferHandle {
    const size_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = 4 * @sizeOf(f32),
    });
    const size_data = [_]f32{ size.min_x, size.min_y, size.max_x, size.max_y, };
    gctx.queue.writeBuffer(gctx.lookupResource(size_buffer).?, 0, f32, size_data[0..]);
    return size_buffer;
}

pub fn createBindGroup(gctx: *zgpu.GraphicsContext, sim: Simulation, compute_bgl: zgpu.BindGroupLayoutHandle, consumer_buffer: zgpu.BufferHandle, stats_buffer: zgpu.BufferHandle, size_buffer: zgpu.BufferHandle, lines_buffer: zgpu.BufferHandle) zgpu.BindGroupHandle {
    var consumer_bind_group: zgpu.BindGroupHandle = undefined;
    const num_consumers = sim.consumers.items.len;
    const num_lines = sim.lines.items.len;
    consumer_bind_group = gctx.createBindGroup(compute_bgl, &[_]zgpu.BindGroupEntryInfo{
        .{
            .binding = 0,
            .buffer_handle = consumer_buffer,
            .offset = 0,
            .size = num_consumers * @sizeOf(Consumer),
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
            .buffer_handle = lines_buffer,
            .offset = 0,
            .size = num_lines * @sizeOf(Line),
        },
    });
    return consumer_bind_group;
}
