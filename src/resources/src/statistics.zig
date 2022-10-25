const std = @import("std");
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const DemoState = @import("resources.zig").DemoState;

const Self = @This();

num_transactions: array(u32),
second: f32 = 0,
max_stat_recorded: u32 = 0,
num_empty_consumers: array(u32),
num_total_producer_inventory: array(u32),

const StagingBuffer = struct {
    slice: ?[]const u32 = null,
    buffer: wgpu.Buffer = undefined,
};

pub const zero = [3]u32{ 0, 0, 0, };

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

pub fn clear(self: *Self) void {
    self.num_transactions.clearAndFree();
    self.num_empty_consumers.clearAndFree();
    self.num_total_producer_inventory.clearAndFree();
    self.num_transactions.append(0) catch unreachable;
    self.num_empty_consumers.append(0) catch unreachable;
    self.num_total_producer_inventory.append(0) catch unreachable;
    self.second = 0;
    self.max_stat_recorded = 0;
}


pub fn clearStatsBuffer(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    const stats_data = [_]u32{0} ** 100;
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, u32, stats_data[0..zero.len]);
}

pub fn createBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const stats_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .storage = true },
        .size = @sizeOf(u32) * zero.len,
    });
    clearStatsBuffer(gctx, stats_buffer);
    return stats_buffer;
}

pub fn createMappedBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = @sizeOf(u32) * zero.len,
    });
}

pub fn getGPUStatistics(demo: *DemoState) [zero.len]u32 {
    var buf: StagingBuffer = .{
        .slice = null,
        .buffer = demo.gctx.lookupResource(demo.buffers.data.stats_mapped).?,
    };
    buf.buffer.mapAsync(
        .{ .read = true },
        0,
        @sizeOf(u32) * zero.len,
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
    clearStatsBuffer(demo.gctx, demo.buffers.data.stats);
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
