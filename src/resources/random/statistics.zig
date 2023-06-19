const std = @import("std");
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const DemoState = @import("main.zig");
const Wgpu = @import("../wgpu.zig");
const Consumer = @import("../consumer.zig");
const Producer = @import("../producer.zig");
const Self = @This();

num_transactions: array(u32),
second: f32 = 0,
num_empty_consumers: array(u32),
num_total_producer_inventory: array(u32),

const StagingBuffer = struct {
    slice: ?[]const u32 = null,
    buffer: wgpu.Buffer = undefined,
};

const SetStagingBuffer = struct {
    slice: ?[]const u32 = null,
    buffer: wgpu.Buffer = undefined,
};

pub const zero = [1]u32{ 0 };
pub const zeroBuffer = [8]u32{ 0, 0, 0, 0, 0, 0, 0, 0 };

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .num_transactions = array(u32).init(allocator),
        .num_empty_consumers = array(u32).init(allocator),
        .num_total_producer_inventory = array(u32).init(allocator),
    };
}

pub fn deinit(self: *Self) void {
    self.num_transactions.deinit();
    self.num_empty_consumers.deinit();
    self.num_total_producer_inventory.deinit();
}

pub fn update(self: *Self, gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const gpu_stats = getGPUStatistics(demo, gctx);
    self.num_transactions.append(gpu_stats[0]) catch unreachable;

    const consumers = Wgpu.getAll(gctx, Consumer, .{
        .structs = demo.buffers.data.consumer,
        .num_structs = Wgpu.getNumStructs(gctx, Consumer, demo.buffers.data.stats),
    }) catch return;
    var empty_consumers: u32 = 0;
    for (consumers) |c| {
        if (c.inventory == 0) {
            empty_consumers += 1;
        }
    }
    self.num_empty_consumers.append(empty_consumers) catch unreachable;

    const producers = Wgpu.getAll(gctx, Producer, .{
        .structs = demo.buffers.data.producer,
        .num_structs = Wgpu.getNumStructs(gctx, Producer, demo.buffers.data.stats),
    }) catch return;
    var total_inventory: u32 = 0;
    for (producers) |p| {
        total_inventory += @intCast(u32, p.inventory);
    }
    self.num_total_producer_inventory.append(total_inventory) catch unreachable;
}

pub fn clear(self: *Self) void {
    self.num_transactions.clearAndFree();
    self.num_empty_consumers.clearAndFree();
    self.num_total_producer_inventory.clearAndFree();
    self.num_transactions.append(0) catch unreachable;
    self.num_empty_consumers.append(0) catch unreachable;
    self.num_total_producer_inventory.append(0) catch unreachable;
    self.second = 0;
}

pub fn clearStatsBuffer(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    const stats_data = [3]u32{ 0, 0, 0 };
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, u32, stats_data[0..]);
}

pub fn clearNumTransactions(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    const stats_data = [1]u32{ 0 };
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, u32, stats_data[0..]);
}

pub fn setNumConsumers(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, num: u32) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        @sizeOf(u32),
        u32,
        &.{ num },
    );
}

pub fn setNumProducers(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle, num: u32) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        2 * @sizeOf(u32),
        u32,
        &.{ num },
    );
}

pub fn createBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const stats_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .storage = true },
        .size = @sizeOf(u32) * zeroBuffer.len,
    });
    clearStatsBuffer(gctx, stats_buffer);
    return stats_buffer;
}

pub fn createMappedBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = @sizeOf(u32) * zeroBuffer.len,
    });
}

pub fn getNumConsumers(demo: *DemoState, gctx: *zgpu.GraphicsContext) u32 {
    return getGPUStatistics(demo, gctx)[1];
}
pub fn getNumProducers(demo: *DemoState, gctx: *zgpu.GraphicsContext) u32 {
    return getGPUStatistics(demo, gctx)[2];
}

pub fn getGPUStatistics(demo: *DemoState, gctx: *zgpu.GraphicsContext) [zero.len]u32 {
    var buf: StagingBuffer = .{
        .slice = null,
        .buffer = gctx.lookupResource(demo.buffers.data.stats.mapped).?,
    };
    buf.buffer.mapAsync(.{ .read = true }, 0, @sizeOf(u32) * zero.len, buffersMappedCallback, @ptrCast(*anyopaque, &buf));
    wait_loop: while (true) {
        gctx.device.tick();
        if (buf.slice == null) {
            continue :wait_loop;
        }
        break;
    }
    buf.buffer.unmap();
    clearNumTransactions(gctx, demo.buffers.data.stats.data);
    return buf.slice.?[0..zero.len].*;
}

fn buffersMappedCallback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
    const usb = @ptrCast(*StagingBuffer, @alignCast(@sizeOf(usize), userdata));
    std.debug.assert(usb.slice == null);
    if (status == .success) {
        usb.slice = usb.buffer.getConstMappedRange(u32, 0, zero.len).?;
    } else {
        std.debug.print("[zgpu] Failed to map buffer (code: {any})\n", .{status});
    }
}

