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
const AnimatedSpline = Splines.AnimatedSpline;
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
