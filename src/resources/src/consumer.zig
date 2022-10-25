const std = @import("std");
const math = std.math;
const array = std.ArrayList;
const Allocator = std.mem.Allocator;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const Main = @import("resources.zig");
const DemoState = Main.DemoState;
const Wgpu = @import("wgpu.zig");

const Self = @This();

position: [4]f32,
home: [4]f32,
destination: [4]f32,
step_size: [4]f32,
color: [4]f32,
moving_rate: f32,
inventory: u32,
radius: f32,
producer_id: i32,

const max_num_consumers = 10000;
const num_vertices = 20;

pub const Parameter = enum {
    moving_rate,
};

const StagingBuffer = struct {
    slice: ?[]const Self = null,
    buffer: wgpu.Buffer = undefined,
    num_consumers: u32 = 0,
};

pub fn create(sim: Simulation) []Self {
    //Unless array len is > max_num_consumers, we get unresponsive consumers
    var consumers: [max_num_consumers + 100]Self = undefined;
    
    var i: usize = 0;
    while (i < sim.params.num_consumers) {
        const x = @intToFloat(
            f32,
            random.intRangeAtMost(
                i32,
                sim.coordinate_size.min_x,
                sim.coordinate_size.max_x
            )
        );

        const y = @intToFloat(
            f32,
            random.intRangeAtMost(
                i32,
                sim.coordinate_size.min_y,
                sim.coordinate_size.max_y
            )
        );

        consumers[i] = Self{
            .position =     [4]f32{ x, y, 0, 0 },
            .home =         [4]f32{ x, y, 0, 0 },
            .destination =  [4]f32{ x, y, 0, 0 },
            .step_size =    [4]f32{ 0, 0, 0, 0 },
            .color =        [4]f32{ 0, 1, 0, 0 },
            .moving_rate =  sim.params.moving_rate,
            .inventory =    0,
            .radius =       sim.params.consumer_radius,
            .producer_id =  1000,
        };
        i += 1;
    }
    return consumers[0..i];
}

pub fn getAll(demo: *DemoState) []const Self {
    const cb_info = demo.gctx.lookupResourceInfo(demo.buffers.data.consumer) orelse unreachable;
    const num_consumers = @intCast(u32, cb_info.size / @sizeOf(Self));

    var buf: StagingBuffer = .{
        .slice = null,
        .buffer = demo.gctx.lookupResource(demo.buffers.data.consumer_mapped).?,
        .num_consumers = num_consumers,
    };
    buf.buffer.mapAsync(
        .{ .read = true },
        0,
        @sizeOf(Self) * num_consumers,
        buffersMappedCallback,
        @ptrCast(*anyopaque, &buf)
    );
    wait_loop: while (true) {
        demo.gctx.device.tick();
        if (buf.slice == null) {
            continue :wait_loop;
        }
        break;
    }
    buf.buffer.unmap();

    return buf.slice.?[0..num_consumers];

}

fn buffersMappedCallback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
    const usb = @ptrCast(*StagingBuffer, @alignCast(@sizeOf(usize), userdata));
    std.debug.assert(usb.slice == null);
    if (status == .success) {
        usb.slice = usb.buffer.getConstMappedRange(Self, 0, usb.num_consumers).?;
    } else {
        std.debug.print("[zgpu] Failed to map buffer (code: {any})\n", .{status});
    }
}

pub fn setAll(demo: *DemoState, parameter: Parameter) void {
    // Get current consumers data
    const consumers = getAll(demo);

    // Set new production rate to 0
    var new_consumers: [max_num_consumers]Self = undefined;
    const params = demo.sim.params;
    for (consumers) |c, i| {
        new_consumers[i] = c;
        switch (parameter) {
            .moving_rate => {
                new_consumers[i].moving_rate = params.moving_rate;
            },
        }
    }

    // Write to consumers buffer again
    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(demo.buffers.data.consumer).?,
        0,
        Self,
        new_consumers[0..demo.sim.params.num_consumers]
    );
}

pub fn add(demo: *DemoState) void {
    const consumer_buffer = createBuffer(demo.gctx, demo.sim);

    const old_consumers = getAll(demo);
    const num_new_consumers = demo.sim.params.num_consumers - old_consumers.len;

    var sim = demo.sim;
    sim.params.num_consumers = @intCast(u32, num_new_consumers);
    const new_consumers = create(sim);

    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(consumer_buffer).?,
        0,
        Self,
        old_consumers[0..],
    );
    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(consumer_buffer).?,
        @sizeOf(Self) * old_consumers.len,
        Self,
        new_consumers[0..],
    );

    demo.bind_groups.compute = Wgpu.createComputeBindGroup(
        demo.gctx,
        consumer_buffer,
        demo.buffers.data.producer,
        demo.buffers.data.stats
    );

    demo.buffers.data.consumer = consumer_buffer;
    demo.buffers.data.consumer_mapped = createMappedBuffer(demo.gctx, demo.sim);
}

pub fn remove(demo: *DemoState) void {
    const consumer_buffer = createBuffer(demo.gctx, demo.sim);
    const old_consumers = getAll(demo);

    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(consumer_buffer).?,
        0,
        Self,
        old_consumers[0..demo.sim.params.num_consumers],
    );
    demo.bind_groups.compute = Wgpu.createComputeBindGroup(
        demo.gctx,
        consumer_buffer,
        demo.buffers.data.producer,
        demo.buffers.data.stats
    );

    demo.buffers.data.consumer = consumer_buffer;
    demo.buffers.data.consumer_mapped = createMappedBuffer(demo.gctx, demo.sim);
}


// Buffer Setup
pub fn createBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{
            .copy_dst = true,
            .copy_src = true,
            .vertex = true,
            .storage = true
        },
        .size = sim.params.num_consumers * @sizeOf(Self),
    });
}

pub fn generate(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, sim: Simulation) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        Self,
        create(sim)
    );
}

pub fn generateBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    const buf = createBuffer(gctx, sim);
    generate(gctx, buf, sim);

    return buf;
}

pub fn createIndexBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const num_triangles = num_vertices - 1;
    const consumer_index_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .index = true },
        .size = num_triangles * 3 * @sizeOf(u32),
    });
    const consumer_index_data = createIndexData(num_triangles);
    gctx.queue.writeBuffer(
        gctx.lookupResource(consumer_index_buffer).?,
        0,
        u32,
        consumer_index_data[0..]
    );
    return consumer_index_buffer;
}

pub fn createVertexBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    const consumer_vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = num_vertices * @sizeOf(f32) * 3,
    });
    var consumer_vertex_data: [num_vertices][3]f32 = undefined;
    const num_sides = @as(f32, num_vertices - 1);
    const angle = 2 * math.pi / num_sides;
    const radius = sim.params.consumer_radius;
    consumer_vertex_data[0] = [3]f32{ 0, 0, 0, };
    var i: u32 = 1;
    while (i < num_vertices) {
        const current_angle = angle * @intToFloat(f32, i);
        const x = @cos(current_angle) * radius;
        const y = @sin(current_angle) * radius;
        consumer_vertex_data[i] = [3]f32{ x, y, 0 };
        i += 1;
    }
    gctx.queue.writeBuffer(
        gctx.lookupResource(consumer_vertex_buffer).?,
        0,
        [3]f32,
        consumer_vertex_data[0..]
    );
    return consumer_vertex_buffer;
}

pub fn createIndexData(comptime num_triangles: u32) [num_triangles * 3]u32 {
    const vertices = num_triangles * 3;
    var indices: [vertices]u32 = undefined;
    var i: usize = 0;
    while (i < num_triangles) {
        indices[i * 3] = 0;
        indices[i * 3 + 1] = @intCast(u32, i) + 1;
        indices[i * 3 + 2] = @intCast(u32, i) + 2;
        i += 1;
    }
    indices[vertices - 1] = 1;
    return indices;
}

pub fn createMappedBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = sim.params.num_consumers * @sizeOf(Self),
    });
}
