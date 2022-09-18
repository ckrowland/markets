const main = @import("bloodstream.zig");
const DemoState = main.DemoState;
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
const Splines = @import("splines.zig");
const Spline = Splines.Spline;
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

pub fn createBindGroup(gctx: *zgpu.GraphicsContext, sim: Simulation, compute_bgl: zgpu.BindGroupLayoutHandle, consumer_buffer: zgpu.BufferHandle, stats_buffer: zgpu.BufferHandle, size_buffer: zgpu.BufferHandle, splines_buffer: zgpu.BufferHandle) zgpu.BindGroupHandle {
    var consumer_bind_group: zgpu.BindGroupHandle = undefined;
    const num_consumers = sim.consumers.items.len;
    const num_splines = sim.splines.stationary.items.len;
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
            .buffer_handle = splines_buffer,
            .offset = 0,
            .size = num_splines * @sizeOf(Spline),
        },
    });
    return consumer_bind_group;
}

pub fn updateBindGroup(demo: *DemoState) zgpu.BindGroupHandle {
    var consumer_bind_group: zgpu.BindGroupHandle = undefined;
    const num_consumers = demo.sim.consumers.items.len;
    const num_splines = demo.sim.splines.stationary.items.len;
    consumer_bind_group = demo.gctx.createBindGroup(
        demo.compute_bind_group_layout,
        &[_]zgpu.BindGroupEntryInfo{
            .{
                .binding = 0,
                .buffer_handle = demo.consumer_buffer,
                .offset = 0,
                .size = num_consumers * @sizeOf(Consumer),
            },
            .{
                .binding = 1,
                .buffer_handle = demo.stats_buffer,
                .offset = 0,
                .size = @sizeOf(GPUStats),
            },
            .{
                .binding = 2,
                .buffer_handle = demo.size_buffer,
                .offset = 0,
                .size = @sizeOf(CoordinateSize),
            },
            .{
                .binding = 3,
                .buffer_handle = demo.splines_buffer,
                .offset = 0,
                .size = num_splines * @sizeOf(Spline),
            },
        }
    );
    return consumer_bind_group;
}
