const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Allocator = std.mem.Allocator;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu.zig");
const DemoState = @import("main.zig");
const Parameters = DemoState.Parameters;
const CoordinateSize = DemoState.CoordinateSize;

const Self = @This();

position: [4]f32,
color: [4]f32,
production_rate: u32,
inventory: u32,
max_inventory: u32,
len: u32,
queue: [450]u32,
_padding: u64 = 0,

pub const Parameter = enum {
    production_rate,
    supply_shock,
    max_inventory,
};

const max_num_producers = 100;

const StagingBuffer = struct {
    slice: ?[]const Self = null,
    buffer: wgpu.Buffer = undefined,
    num_producers: u32 = 0,
};

pub fn create(params: Parameters, coordinate_size: CoordinateSize) []Self {
    var producers: [max_num_producers]Self = undefined;

    var i: usize = 0;
    while (i < params.num_producers) {
        const x = @intToFloat(f32, random.intRangeAtMost(i32, coordinate_size.min_x, coordinate_size.max_x));

        const y = @intToFloat(f32, random.intRangeAtMost(i32, coordinate_size.min_y, coordinate_size.max_y));

        producers[i] = Self{
            .position = [4]f32{ x, y, 0, 0 },
            .color = [4]f32{ 1, 1, 1, 0 },
            .production_rate = params.production_rate,
            .inventory = params.max_inventory,
            .max_inventory = params.max_inventory,
            .len = 0,
            .queue = [_]u32{0} ** 450,
        };
        i += 1;
    }
    return producers[0..i];
}

pub fn getAll(demo: *DemoState, gctx: *zgpu.GraphicsContext) []const Self {
    const pb_info = gctx.lookupResourceInfo(demo.buffers.data.producer) orelse unreachable;
    const num_producers = @intCast(u32, pb_info.size / @sizeOf(Self));

    var buf: StagingBuffer = .{
        .slice = null,
        .buffer = gctx.lookupResource(demo.buffers.data.producer_mapped).?,
        .num_producers = num_producers,
    };
    buf.buffer.mapAsync(.{ .read = true }, 0, @sizeOf(Self) * num_producers, buffersMappedCallback, @ptrCast(*anyopaque, &buf));
    wait_loop: while (true) {
        gctx.device.tick();
        if (buf.slice == null) {
            continue :wait_loop;
        }
        break;
    }
    buf.buffer.unmap();

    return buf.slice.?[0..num_producers];
}

pub fn add(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    const producer_buffer = createBuffer(gctx, demo.params.num_producers);

    const old_producers = getAll(demo, gctx);
    const num_new_producers = demo.params.num_producers - old_producers.len;

    var params = demo.params;
    params.num_producers = @intCast(u32, num_new_producers);
    const new_producers = create(params, demo.coordinate_size);

    gctx.queue.writeBuffer(
        gctx.lookupResource(producer_buffer).?,
        0,
        Self,
        old_producers[0..],
    );
    gctx.queue.writeBuffer(
        gctx.lookupResource(producer_buffer).?,
        @sizeOf(Self) * old_producers.len,
        Self,
        new_producers[0..],
    );

    demo.bind_groups.compute = Wgpu.createComputeBindGroup(gctx, demo.buffers.data.consumer, producer_buffer, demo.buffers.data.stats);

    demo.buffers.data.producer = producer_buffer;
    demo.buffers.data.producer_mapped = Wgpu.createMappedBuffer(gctx, Self, demo.params.num_producers);
}

pub fn remove(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    const producer_buffer = createBuffer(gctx, demo.params.num_producers);
    const old_producers = getAll(demo, gctx);

    gctx.queue.writeBuffer(
        gctx.lookupResource(producer_buffer).?,
        0,
        Self,
        old_producers[0..demo.params.num_producers],
    );
    demo.bind_groups.compute = Wgpu.createComputeBindGroup(gctx, demo.buffers.data.consumer, producer_buffer, demo.buffers.data.stats);

    demo.buffers.data.producer = producer_buffer;
    demo.buffers.data.producer_mapped = Wgpu.createMappedBuffer(gctx, Self, demo.params.num_producers);
}

pub fn setAll(demo: *DemoState, gctx: *zgpu.GraphicsContext, parameter: Parameter) void {
    // Get current producers data
    const producers = getAll(demo, gctx);

    // Set new production rate to 0
    var new_producers: [max_num_producers]Self = undefined;
    const params = demo.params;
    for (producers) |p, i| {
        new_producers[i] = p;
        switch (parameter) {
            .production_rate => {
                new_producers[i].production_rate = params.production_rate;
            },
            .supply_shock => {
                new_producers[i].inventory = 0;
            },
            .max_inventory => {
                new_producers[i].max_inventory = params.max_inventory;
            },
        }
    }

    // Write to producers buffer again
    gctx.queue.writeBuffer(gctx.lookupResource(demo.buffers.data.producer).?, 0, Self, new_producers[0..demo.params.num_producers]);
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

pub fn createBuffer(gctx: *zgpu.GraphicsContext, num_producers: u32) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .vertex = true, .storage = true },
        .size = num_producers * @sizeOf(Self),
    });
}

pub fn generate(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, params: Parameters, coordinate_size: CoordinateSize) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        Self,
        create(params, coordinate_size),
    );
}

pub fn generateBuffer(gctx: *zgpu.GraphicsContext, params: Parameters, coordinate_size: CoordinateSize) zgpu.BufferHandle {
    const buf = createBuffer(gctx, params.num_producers);
    generate(gctx, buf, params, coordinate_size);

    return buf;
}
