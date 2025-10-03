const std = @import("std");
const zgpu = @import("zgpu");
const Wgpu = @import("wgpu");
const Consumer = @import("consumer");
const Producer = @import("producer");
const Camera = @import("camera");

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

//pub fn producerMitosis(args: Wgpu.CallbackArgs(Producer)) void {
//    const slice = args.obj_buf.mapping.staging.slice;
//    if (slice) |producers| {
//        for (producers) |p| {
//            if (p.money >= p.max_money) {
//                const new_p = Producer.
//                //create new producer
//                //add to producer buffer
//            }
//        }
//    }
//}

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
        for (producers, 0..) |p, i| {
            const new_coord = Camera.getWorldPosition(args.gctx, p.absolute_home);
            const offset = i * @sizeOf(Producer) + @offsetOf(Producer, "home");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});
        }
    }
}

pub fn updateConsumerCoords(args: Wgpu.CallbackArgs(Consumer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    const buf = args.gctx.lookupResource(args.obj_buf.buf).?;

    if (slice) |consumers| {
        for (consumers, 0..) |c, i| {
            const new_coord = Camera.getWorldPosition(args.gctx, c.absolute_home);
            var offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "home");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});

            offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "position");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});

            offset = i * @sizeOf(Consumer) + @offsetOf(Consumer, "destination");
            args.gctx.queue.writeBuffer(buf, offset, [4]f32, &.{new_coord});
        }
    }
}
