const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const Wgpu = @import("libs/wgpu.zig");
const Consumer = @import("libs/consumer.zig");
const Producer = @import("libs/producer.zig");
const Camera = @import("libs/camera.zig");

pub fn getNumAgents(args: Wgpu.CallbackArgs(u32)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    if (slice) |stats| {
        args.consumer_num_structs.* = stats[1];
        args.producer_num_structs.* = stats[2];
        args.gui_slider.* = stats[2];
    }
}

pub fn price(args: Wgpu.CallbackArgs(Producer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    var avg: u32 = 0;
    if (slice) |producers| {
        for (producers) |p| {
            avg += p.price;
        }
        avg /= @intCast(producers.len);
    }
    //args.stat_array.append(avg) catch unreachable;
    args.gui_slider.* = avg;
}

pub fn numTransactions(args: Wgpu.CallbackArgs(u32)) void {
    const slice = args.obj_buf.mapping.staging.slice;

    var num: u32 = 0;
    if (slice) |stats| {
        num = stats[0];
    }
    args.stat_array.append(num) catch unreachable;

    const resource = args.gctx.lookupResource(args.obj_buf.buf).?;
    args.gctx.queue.writeBuffer(resource, 0, u32, &.{0});
}

pub fn avgProducerInventory(args: Wgpu.CallbackArgs(Producer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    var avg_inventory: u32 = 0;
    if (slice) |producers| {
        for (producers) |p| {
            avg_inventory += p.inventory;
        }
        avg_inventory /= @intCast(producers.len);
    }
    args.stat_array.append(avg_inventory) catch unreachable;
}

pub fn avgProducerMoney(args: Wgpu.CallbackArgs(Producer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    var avg_money: u32 = 0;
    if (slice) |producers| {
        for (producers) |p| {
            avg_money += p.money;
        }
        avg_money /= @intCast(producers.len);
    }
    args.stat_array.append(avg_money) catch unreachable;
}

pub fn avgConsumerInventory(args: Wgpu.CallbackArgs(Consumer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    var avg_inventory: u32 = 0;
    if (slice) |consumers| {
        for (consumers) |c| {
            avg_inventory += c.inventory;
        }
        avg_inventory /= @intCast(consumers.len);
    }
    args.stat_array.append(avg_inventory) catch unreachable;
}

pub fn avgConsumerMoney(args: Wgpu.CallbackArgs(Consumer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    var avg_money: u32 = 0;
    if (slice) |consumers| {
        for (consumers) |c| {
            avg_money += c.money;
        }
        avg_money /= @intCast(consumers.len);
    }
    args.stat_array.append(avg_money) catch unreachable;
}

pub fn emptyConsumers(args: Wgpu.CallbackArgs(Consumer)) void {
    const consumers = Wgpu.getMappedData(Consumer, &args.obj_buf.mapping);
    var empty_consumers: u32 = 0;
    for (consumers) |c| {
        if (c.inventory == 0) {
            empty_consumers += 1;
        }
    }
    args.stat_array.append(empty_consumers) catch unreachable;
}

pub fn updateProducerCoords(args: Wgpu.CallbackArgs(Producer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    const buf = args.gctx.lookupResource(args.obj_buf.buf).?;

    if (slice) |producers| {
        Camera.updateMaxX(args.gctx);
        for (producers, 0..) |p, i| {
            const new_coord = Camera.updatePosition(p.home);
            const offset = i * @sizeOf(Producer) + @offsetOf(Producer, "home");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});
        }
    }

    args.gctx.queue.writeBuffer(
        args.gctx.lookupResource(args.stat_buf.buf).?,
        4 * @sizeOf(u32),
        u32,
        &.{ Camera.MAX_X, Camera.MAX_Y },
    );
}

pub fn updateConsumerCoords(args: Wgpu.CallbackArgs(Consumer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    const buf = args.gctx.lookupResource(args.obj_buf.buf).?;

    if (slice) |consumers| {
        for (consumers, 0..) |c, i| {
            const new_coord = Camera.updatePosition(c.home);
            var offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "home");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});

            offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "position");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});

            offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "destination");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});
        }
    }
}

pub fn updateConsumerMoney(args: Wgpu.CallbackArgs(Consumer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    const buf = args.gctx.lookupResource(args.obj_buf.buf).?;

    if (slice) |consumers| {
        for (consumers, 0..) |c, i| {
            if (c.money > c.max_money) {
                const offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "money");
                args.gctx.queue.writeBuffer(buf, offset, u32, &.{c.max_money});
            }
        }
    }
}

pub fn updateProducerMoney(args: Wgpu.CallbackArgs(Producer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    const buf = args.gctx.lookupResource(args.obj_buf.buf).?;

    if (slice) |producers| {
        for (producers, 0..) |p, i| {
            if (p.money > p.max_money) {
                const offset = i * @sizeOf(Producer) + @offsetOf(Producer, "money");
                args.gctx.queue.writeBuffer(buf, offset, u32, &.{p.max_money});
            }
        }
    }
}

pub fn updateProducerInventory(args: Wgpu.CallbackArgs(Producer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    const buf = args.gctx.lookupResource(args.obj_buf.buf).?;

    if (slice) |producers| {
        for (producers, 0..) |p, i| {
            if (p.inventory > p.max_inventory) {
                const offset = i * @sizeOf(Producer) + @offsetOf(Producer, "inventory");
                args.gctx.queue.writeBuffer(buf, offset, u32, &.{p.max_inventory});
            }
        }
    }
}
