const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Allocator = std.mem.Allocator;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu.zig");
const Simulation = @import("simulation.zig");
const Main = @import("resources.zig");
const DemoState = Main.DemoState;

const Self = @This();

position: [4]f32,
color: [4]f32,
production_rate: u32,
giving_rate: u32,
inventory: u32,
max_inventory: u32,
len: u32,
queue: [450]u32,
_padding: u32 = 0,

pub const Parameter = enum {
    production_rate,
    giving_rate,
    inventory,
    max_inventory,
};

const max_num_producers = 100;

const StagingBuffer = struct {
    slice: ?[]const Self = null,
    buffer: wgpu.Buffer = undefined,
    num_producers: u32 = 0,
};


pub fn createProducers(sim: Simulation) []Self {
    var producers: [max_num_producers]Self = undefined;

    var i: usize = 0;
    while (i < sim.params.num_producers) {
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

        producers[i] = Self{
            .position = [4]f32{ x, y, 0, 0 },
            .color = [4]f32{ 1, 1, 1, 0 },
            .production_rate = sim.params.production_rate,
            .giving_rate = sim.params.giving_rate,
            .inventory = sim.params.max_inventory,
            .max_inventory = sim.params.max_inventory,
            .len = 0,
            .queue = [_]u32{0} ** 450,
        };
        i += 1;
    }
    return producers[0..i];
}

pub fn getProducers(demo: *DemoState) []const Self {
    const pb_info = demo.gctx.lookupResourceInfo(demo.buffers.data.producer) orelse unreachable;
    const num_producers = @intCast(u32, pb_info.size / @sizeOf(Self));

    var buf: StagingBuffer = .{
        .slice = null,
        .buffer = demo.gctx.lookupResource(demo.buffers.data.producer_mapped).?,
        .num_producers = num_producers,
    };
    buf.buffer.mapAsync(
        .{ .read = true },
        0,
        @sizeOf(Self) * num_producers,
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

    return buf.slice.?[0..num_producers];

}

pub fn addProducers(demo: *DemoState) void {
    const producers = createBuffer(demo.gctx, demo.sim);

    const old_producers = getProducers(demo);
    const num_new_producers = demo.sim.params.num_producers - old_producers.len;

    var sim = demo.sim;
    sim.params.num_producers = @intCast(u32, num_new_producers);
    const new_producers = createProducers(sim);

    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(producers).?,
        0,
        Self,
        old_producers[0..],
    );
    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(producers).?,
        @sizeOf(Self) * old_producers.len,
        Self,
        new_producers[0..],
    );

    demo.bind_groups.compute = Wgpu.createComputeBindGroup(
        demo.gctx,
        demo.buffers.data.consumer,
        producers,
        demo.buffers.data.stats
    );

    demo.buffers.data.producer = producers;
    demo.buffers.data.producer_mapped = createMappedBuffer(demo.gctx, demo.sim);
}

pub fn removeProducers(demo: *DemoState) void {
    const producer_buffer = createBuffer(demo.gctx, demo.sim);
    const old_producers = getProducers(demo);

    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(producer_buffer).?,
        0,
        Self,
        old_producers[0..demo.sim.params.num_producers],
    );
    demo.bind_groups.compute = Wgpu.createComputeBindGroup(
        demo.gctx,
        demo.buffers.data.consumer,
        producer_buffer,
        demo.buffers.data.stats
    );

    demo.buffers.data.producer = producer_buffer;
    demo.buffers.data.producer_mapped = createMappedBuffer(demo.gctx, demo.sim);
}

pub fn setParameterAll(demo: *DemoState, parameter: Parameter) void {
    // Get current producers data
    const producers = getProducers(demo);

    // Set new production rate to 0
    var new_producers: [max_num_producers]Self = undefined;
    const params = demo.sim.params;
    for (producers) |p, i| {
        new_producers[i] = p;
        switch (parameter) {
            .production_rate => {
                new_producers[i].production_rate = params.production_rate;
            },
            .giving_rate => {
                new_producers[i].giving_rate = params.giving_rate;
            },
            .inventory => {
                new_producers[i].inventory = 0;
            },
            .max_inventory => {
                new_producers[i].max_inventory = params.max_inventory;
            },
        }
    }

    // Write to producers buffer again
    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(demo.buffers.data.producer).?,
        0,
        Self,
        new_producers[0..demo.sim.params.num_producers]
    );
}

fn buffersMappedCallback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
    const usb = @ptrCast(*StagingBuffer, @alignCast(@sizeOf(usize), userdata));
    std.debug.assert(usb.slice == null);
    if (status == .success) {
        usb.slice = usb.buffer.getConstMappedRange(Self, 0, usb.num_producers).?;
    } else {
        std.debug.print("[zgpu] Failed to map buffer (code: {any})\n", .{status});
    }
}

pub fn createBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{
            .copy_dst = true,
            .copy_src = true,
            .vertex = true,
            .storage = true
        },
        .size = sim.params.num_producers * @sizeOf(Self),
    });
}

pub fn generateProducers(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, sim: Simulation) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        Self,
        createProducers(sim),
    );
}

pub fn generateBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    const buf = createBuffer(gctx, sim);
    generateProducers(gctx, buf, sim);

    return buf;
}

pub fn createMappedBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = sim.params.num_producers * @sizeOf(Self),
    });
}

pub fn createVertexBuffer(gctx: *zgpu.GraphicsContext, sim: Simulation) zgpu.BufferHandle {
    const producer_vertex_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = 6 * @sizeOf(f32) * 3,
    });

    const width = sim.params.producer_width;
    const upper_left = [3]f32{ -width, width, 0.0 };
    const lower_left = [3]f32{ -width, -width, 0.0 };
    const upper_right = [3]f32{ width, width, 0.0 };
    const lower_right = [3]f32{ width, -width, 0.0 };

    const vertex_array = [6][3]f32{
        upper_left,
        lower_left,
        lower_right,
        lower_right,
        upper_right,
        upper_left,
    };

    gctx.queue.writeBuffer(gctx.lookupResource(producer_vertex_buffer).?, 0, [3]f32, vertex_array[0..]);
    return producer_vertex_buffer;
}
