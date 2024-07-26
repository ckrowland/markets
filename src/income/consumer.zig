const std = @import("std");
const math = std.math;
const array = std.ArrayList;
const Allocator = std.mem.Allocator;
const random = std.crypto.random;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Parameters = Main.Parameters;
const Wgpu = @import("wgpu.zig");
const Camera = @import("camera.zig");
const Statistics = @import("statistics.zig");
const Self = @This();

absolute_home: [4]i32 = .{ 0, 0, 0, 0 },
position: [4]f32 = .{ 0, 0, 0, 0 },
home: [4]f32 = .{ 0, 0, 0, 0 },
destination: [4]f32 = .{ 0, 0, 0, 0 },
color: [4]f32 = .{ 1, 0, 0, 0 },
step_size: [2]f32 = .{ 0, 0 },
moving_rate: f32 = 0,
max_demand_rate: u32 = 0,
income_quartile: u32 = 0,
income: u32 = 0,
radius: f32 = 20.0,
inventory: u32 = 0,
balance: u32 = 0,
max_balance: u32 = 100000,
producer_id: i32 = -1,
grouping_id: u32 = 0,

pub const z_pos = 0;
pub fn generateFromParams(demo: *DemoState) void {
    const consumer_incomes = demo.params.consumer_incomes;
    for (consumer_incomes, 0..) |ci, i| {
        const num: u32 = @intFromFloat(ci.new.num);
        for (0..num) |_| {
            const c = createNewConsumer(demo, @intCast(i));
            appendConsumer(demo, c);
        }
    }
}

pub fn createNewConsumer(demo: *DemoState, i: u32) Self {
    const x = random.intRangeAtMost(i32, Camera.MIN_X, Camera.MAX_X);
    const y = random.intRangeAtMost(i32, Camera.MIN_Y, Camera.MAX_Y);
    const f_x = @as(f32, @floatFromInt(x)) * demo.params.aspect;
    const f_y = @as(f32, @floatFromInt(y));
    const home = [4]f32{ f_x, f_y, z_pos, 1 };
    return .{
        .absolute_home = .{ x, y, z_pos, 1 },
        .position = home,
        .home = home,
        .destination = home,
        .income = @intFromFloat(demo.params.consumer_incomes[i].new.income),
        .income_quartile = i,
        .moving_rate = demo.params.moving_rate,
        .max_demand_rate = demo.params.max_demand_rate,
    };
}

pub fn appendConsumer(demo: *DemoState, c: Self) void {
    const obj_buf = &demo.buffers.data.consumers;
    var consumers: [1]Self = .{c};
    Wgpu.appendBuffer(demo.gctx, Self, .{
        .num_old_structs = @as(u32, @intCast(obj_buf.list.items.len)),
        .buf = obj_buf.buf,
        .structs = consumers[0..],
    });
    obj_buf.list.append(c) catch unreachable;
    obj_buf.mapping.num_structs += 1;
    demo.params.consumer_incomes[c.income_quartile].num += 1;
}

pub fn setParamAll(
    demo: *DemoState,
    comptime tag: []const u8,
    comptime T: type,
    num: T,
) void {
    const buf = demo.buffers.data.consumers.buf;
    const resource = demo.gctx.lookupResource(buf).?;
    const field_enum = @field(std.meta.FieldEnum(Self), tag);
    const field_type = std.meta.FieldType(Self, field_enum);
    std.debug.assert(field_type == T);

    const struct_offset = @offsetOf(Self, tag);
    for (demo.buffers.data.consumers.list.items, 0..) |_, i| {
        const offset = i * @sizeOf(Self) + struct_offset;
        demo.gctx.queue.writeBuffer(resource, offset, field_type, &.{num});
    }
}

pub fn setQuartileIncome(demo: *DemoState, quartile: u32, income: u32) void {
    const buf = demo.buffers.data.consumers.buf;
    const resource = demo.gctx.lookupResource(buf).?;
    const struct_offset = @offsetOf(Self, "income");

    for (demo.buffers.data.consumers.list.items, 0..) |c, i| {
        if (c.income_quartile == quartile) {
            const offset = i * @sizeOf(Self) + struct_offset;
            demo.gctx.queue.writeBuffer(resource, offset, u32, &.{income});
        }
    }
}

pub fn expandQuartile(demo: *DemoState, quartile: u32, num: u32) void {
    const num_to_add: usize = @intCast(num);
    for (0..num_to_add) |_| {
        const c = createNewConsumer(demo, quartile);
        appendConsumer(demo, c);
    }
}

pub fn shrinkQuartile(demo: *DemoState, quartile: u32, num: u32) void {
    const num_to_remove: usize = @intCast(num);
    for (0..num_to_remove) |_| {
        removeOneFromQuartile(demo, quartile) catch unreachable;
    }
}

pub fn removeOneFromQuartile(demo: *DemoState, quartile: u32) !void {
    const buf = &demo.buffers.data.consumers;
    for (buf.list.items, 0..) |c, i| {
        if (c.income_quartile == quartile) {
            replaceIndexWithLastElement(demo, i);
            zeroOutLastElement(demo);

            _ = buf.list.swapRemove(i);
            buf.mapping.num_structs -= 1;
            demo.params.consumer_incomes[quartile].num -= 1;
            return;
        }
    }
    return error.CouldNotRemove;
}

pub fn replaceIndexWithLastElement(demo: *DemoState, i: usize) void {
    const buf = demo.buffers.data.consumers.buf;
    const buff = demo.gctx.lookupResource(buf).?;
    const offset = @sizeOf(Self) * i;
    const consumers = demo.buffers.data.consumers;
    const end_index = consumers.list.items.len - 1;
    const replace = consumers.list.items[end_index];
    demo.gctx.queue.writeBuffer(buff, offset, Self, &.{replace});
}

pub fn zeroOutLastElement(demo: *DemoState) void {
    const buf = demo.buffers.data.consumers.buf;
    const buff = demo.gctx.lookupResource(buf).?;
    const end_index = demo.buffers.data.consumers.list.items.len - 1;
    const end_offset = @sizeOf(Self) * end_index;
    const all_zero = [_]u8{0} ** @sizeOf(Self);
    const zeros = all_zero[0..];
    demo.gctx.queue.writeBuffer(buff, end_offset, u8, zeros);
}
