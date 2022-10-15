const std = @import("std");
const zgpu = @import("zgpu");

pub const array = [3]u32{ 0, 0, 0, };

pub fn clearStatsBuffer(gctx: *zgpu.GraphicsContext, buf: zgpu.BufferHandle) void {
    const stats_data = [_]u32{0} ** 100;
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, u32, stats_data[0..array.len]);
}

pub fn createStatsBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const stats_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .storage = true },
        .size = @sizeOf(u32) * array.len,
    });
    clearStatsBuffer(gctx, stats_buffer);
    return stats_buffer;
}

pub fn createStatsMappedBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = @sizeOf(u32) * array.len,
    });
}
