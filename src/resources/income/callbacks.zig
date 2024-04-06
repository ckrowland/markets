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
    const slice = args.buf.mapping.staging.slice;

    var num: u32 = 0;
    if (slice) |stats| {
        num = stats[0];
    }
    args.stats.num_transactions.append(num) catch unreachable;
    Statistics.clearNumTransactions(args.gctx, args.buf.buf);
}

pub fn totalInventory(args: Args(Producer)) void {
    const slice = args.buf.mapping.staging.slice;

    var total_inventory: u32 = 0;
    if (slice) |producers| {
        for (producers) |p| {
            total_inventory += @as(u32, @intCast(p.inventory));
        }
    }
    args.stats.num_total_producer_inventory.append(total_inventory) catch unreachable;
}

pub fn consumerStats(args: Args(Consumer)) void {
    const slice = args.buf.mapping.staging.slice;

    var empty_consumers: u32 = 0;
    if (slice) |consumers| {
        for (consumers) |c| {
            if (c.inventory == 0) {
                empty_consumers += 1;
            }
        }
    }
    args.stats.num_empty_consumers.append(empty_consumers) catch unreachable;

    var total_balance: u32 = 0;
    if (slice) |consumers| {
        for (consumers) |c| {
            total_balance += @intFromFloat(c.balance);
        }
    }
    const len: u32 = @intCast(args.stats.avg_consumer_balance.items.len + 1);
    const avg_balance = total_balance / len;

    args.stats.avg_consumer_balance.append(avg_balance) catch unreachable;
}
