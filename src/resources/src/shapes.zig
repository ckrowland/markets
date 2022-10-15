const main = @import("resources.zig");
const Vertex = main.Vertex;
const Statistics = @import("statistics.zig");
const std = @import("std");
const math = std.math;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const Consumer = Simulation.Consumer;
const Producer = Simulation.Producer;
const wgsl = @import("shaders.zig");
const array = std.ArrayList;
const Wgpu = @import("wgpu.zig");


pub fn createBindGroup(gctx: *zgpu.GraphicsContext, sim: Simulation, consumer_buffer: zgpu.BufferHandle, producer_buffer: zgpu.BufferHandle, stats_buffer: zgpu.BufferHandle) zgpu.BindGroupHandle {
    const compute_bgl = Wgpu.createComputeBindGroupLayout(gctx);
    defer gctx.releaseResource(compute_bgl);

    return gctx.createBindGroup(compute_bgl, &[_]zgpu.BindGroupEntryInfo{
        .{
            .binding = 0,
            .buffer_handle = consumer_buffer,
            .offset = 0,
            .size = sim.consumers.items.len * @sizeOf(Consumer),
        },
        .{
            .binding = 1,
            .buffer_handle = producer_buffer,
            .offset = 0,
            .size = sim.producers.items.len * @sizeOf(Producer),
        },
        .{
            .binding = 2,
            .buffer_handle = stats_buffer,
            .offset = 0,
            .size = @sizeOf(u32) * Statistics.array.len,
        },
    });
}
