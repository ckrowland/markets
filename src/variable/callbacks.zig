const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Camera = @import("camera.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Statistics = @import("statistics.zig");
const Wgpu = @import("wgpu.zig");

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
            total_balance += c.balance;
        }
    }
    const len: u32 = @intCast(args.buf.mapping.num_structs + 1);
    const avg_balance = total_balance / len;
    args.stats.avg_consumer_balance.append(avg_balance) catch unreachable;
}

pub fn producerStats(args: Args(Producer)) void {
    const slice = args.buf.mapping.staging.slice;

    var total_inventory: u32 = 0;
    var total_balance: u32 = 0;
    var total_margin: u32 = 0;
    if (slice) |producers| {
        const buf = args.gctx.lookupResource(args.buf.buf).?;
        const pns_offset = @offsetOf(Producer.Params, "prev_num_sales");
        const ns_offset = @offsetOf(Producer.Params, "num_sales");
        const ma_offset = @offsetOf(Producer.Params, "margin");
        for (producers, 0..) |p, i| {
            total_inventory += @intCast(p.inventory);
            total_balance += @intCast(p.balance);
            total_margin += @intCast(p.params.margin);

            var margin = p.params.margin;
            const num_sales = p.params.num_sales;
            if (num_sales < p.params.prev_num_sales and margin > 10) {
                margin -= 10;
            } else if (num_sales > p.params.prev_num_sales) {
                margin += 10;
            }

            const index_offset = i * @sizeOf(Producer);
            const p_offset = index_offset + pns_offset;
            args.gctx.queue.writeBuffer(buf, p_offset, u32, &.{num_sales});

            const n_offset = index_offset + ns_offset;
            args.gctx.queue.writeBuffer(buf, n_offset, u32, &.{0});

            const m_offset = index_offset + ma_offset;
            args.gctx.queue.writeBuffer(buf, m_offset, u32, &.{margin});
        }
    }
    args.stats.num_total_producer_inventory.append(total_inventory) catch unreachable;

    const len: u32 = @intCast(args.buf.mapping.num_structs + 1);
    const avg_balance = total_balance / len;
    args.stats.avg_producer_balance.append(avg_balance) catch unreachable;

    const avg_margin = total_margin / len;
    args.stats.avg_margin.append(avg_margin) catch unreachable;
}

pub fn updateProducerCoords(args: Args(Producer)) void {
    const slice = args.buf.mapping.staging.slice;

    if (slice) |producers| {
        for (producers, 0..) |p, i| {
            const new_coord = Camera.getWorldPosition(args.gctx, p.params.absolute_home);
            const buf = args.gctx.lookupResource(args.buf.buf).?;
            const offset = i * @sizeOf(Producer) + @offsetOf(Producer.Params, "home");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});
        }
    }
}

pub fn updateConsumerCoords(args: Args(Consumer)) void {
    const slice = args.buf.mapping.staging.slice;

    if (slice) |consumers| {
        for (consumers, 0..) |c, i| {
            const new_coord = Camera.getWorldPosition(args.gctx, c.absolute_home);
            const buf = args.gctx.lookupResource(args.buf.buf).?;
            var offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "home");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});

            offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "position");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});

            offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "destination");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});
        }
    }
}
