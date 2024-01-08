const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Statistics = @import("statistics.zig");

pub fn Args(comptime T: type) type {
    return struct {
        gctx: *zgpu.GraphicsContext,
        buf: *Wgpu.ObjectBuffer(T),
        stats: *Statistics = undefined,
    };
}

pub fn numTransactions(args: Args(u32)) void {
    const gpu_stats = Wgpu.getMappedData(u32, &args.buf.mapping);
    args.stats.num_transactions.append(gpu_stats[0]) catch unreachable;
    Statistics.clearNumTransactions(args.gctx, args.buf.buf);
}

pub fn totalInventory(args: Args(Producer)) void {
    const producers = Wgpu.getMappedData(Producer, &args.buf.mapping);
    var total_inventory: u32 = 0;
    for (producers) |p| {
        total_inventory += @as(u32, @intCast(p.inventory));
    }
    args.stats.num_total_producer_inventory.append(total_inventory) catch unreachable;
}

pub fn emptyConsumers(args: Args(Consumer)) void {
    const consumers = Wgpu.getMappedData(Consumer, &args.buf.mapping);
    var empty_consumers: u32 = 0;
    for (consumers) |c| {
        if (c.inventory == 0) {
            empty_consumers += 1;
        }
    }
    args.stats.num_empty_consumers.append(empty_consumers) catch unreachable;
}
