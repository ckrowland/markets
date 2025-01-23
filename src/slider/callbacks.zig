const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Wgpu = @import("wgpu");
const Consumer = @import("consumer");
const Producer = @import("producer");
const Statistics = @import("statistics");
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

pub fn totalInventory(args: Wgpu.CallbackArgs(Producer)) void {
    const slice = args.obj_buf.mapping.staging.slice;
    var total_inventory: u32 = 0;
    if (slice) |producers| {
        for (producers) |p| {
            total_inventory += @intCast(p.inventory);
        }
    }
    args.stat_array.append(total_inventory) catch unreachable;
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
