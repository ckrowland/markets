const std = @import("std");
const math = std.math;
const array = std.ArrayList;
const Allocator = std.mem.Allocator;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Main = @import("resources.zig");
const DemoState = Main.DemoState;
const Parameters = Main.Parameters;
const CoordinateSize = Main.CoordinateSize;
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

pub fn create(params: Parameters, coordinate_size: CoordinateSize) []Self {
    //Unless array len is > max_num_consumers, we get unresponsive consumers
    var consumers: [max_num_consumers + 100]Self = undefined;
    
    var i: usize = 0;
    while (i < params.num_consumers) {
        const x = @intToFloat(
            f32,
            random.intRangeAtMost(
                i32,
                coordinate_size.min_x,
                coordinate_size.max_x
            )
        );

        const y = @intToFloat(
            f32,
            random.intRangeAtMost(
                i32,
                coordinate_size.min_y,
                coordinate_size.max_y
            )
        );

        consumers[i] = Self{
            .position =     [4]f32{ x, y, 0, 0 },
            .home =         [4]f32{ x, y, 0, 0 },
            .destination =  [4]f32{ x, y, 0, 0 },
            .step_size =    [4]f32{ 0, 0, 0, 0 },
            .color =        [4]f32{ 0, 1, 0, 0 },
            .moving_rate =  params.moving_rate,
            .inventory =    0,
            .radius =       params.consumer_radius,
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
    const params = demo.params;
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
        new_consumers[0..demo.params.num_consumers]
    );
}

pub fn add(demo: *DemoState) void {
    const consumer_buffer = Wgpu.createBuffer(
        demo.gctx,
        Self,
        demo.params.num_consumers
    );

    const old_consumers = getAll(demo);
    const num_new_consumers = demo.params.num_consumers - old_consumers.len;

    var params = demo.params;
    params.num_consumers = @intCast(u32, num_new_consumers);
    const new_consumers = create(params, demo.coordinate_size);

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
    demo.buffers.data.consumer_mapped = Wgpu.createMappedBuffer(
        demo.gctx,
        Self,
        demo.params.num_consumers
    );
}

pub fn remove(demo: *DemoState) void {
    const consumer_buffer = Wgpu.createBuffer(
        demo.gctx,
        Self,
        demo.params.num_consumers
    );
    const old_consumers = getAll(demo);

    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(consumer_buffer).?,
        0,
        Self,
        old_consumers[0..demo.params.num_consumers],
    );
    demo.bind_groups.compute = Wgpu.createComputeBindGroup(
        demo.gctx,
        consumer_buffer,
        demo.buffers.data.producer,
        demo.buffers.data.stats
    );

    demo.buffers.data.consumer = consumer_buffer;
    demo.buffers.data.consumer_mapped = Wgpu.createMappedBuffer(
        demo.gctx,
        Self,
        demo.params.num_consumers
    );
}


// Buffer Setup
pub fn generate(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    params: Parameters,
    coordinate_size: CoordinateSize
) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        Self,
        create(params, coordinate_size)
    );
}

pub fn generateBuffer(
    gctx: *zgpu.GraphicsContext,
    params: Parameters,
    coordinate_size: CoordinateSize
) zgpu.BufferHandle {

    const buf = Wgpu.createBuffer(gctx, Self, params.num_consumers);
    generate(gctx, buf, params, coordinate_size);

    return buf;
}
