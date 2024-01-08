const std = @import("std");
const zgpu = @import("zgpu");
const Wgpu = @import("../wgpu.zig");
const Consumer = @import("../consumer.zig");

const Self = @This();

absolute_home: [4]i32,
home: [4]f32,
color: [4]f32 = .{ 0, 0, 0, 0 },
grouping_id: u32 = 0,

pub const z_pos = 0;

pub fn create(args: Consumer.Args) Self {
    return Self{
        .absolute_home = .{ args.absolute_home[0], args.absolute_home[1], z_pos, 1 },
        .home = .{ args.home[0], args.home[1], z_pos, 1 },
        .grouping_id = args.grouping_id,
    };
}

pub const AppendArgs = struct {
    args: Consumer.Args,
    buf: *Wgpu.ObjectBuffer(Self),
};
pub fn createAndAppend(gctx: *zgpu.GraphicsContext, args: AppendArgs) void {
    var hovers: [1]Self = .{
        create(args.args),
    };
    Wgpu.appendBuffer(gctx, Self, .{
        .num_old_structs = @as(u32, @intCast(args.buf.list.items.len)),
        .buf = args.buf.buf,
        .structs = hovers[0..],
    });
    args.buf.list.append(hovers[0]) catch unreachable;
}

pub fn highlightConsumers(
    gctx: *zgpu.GraphicsContext,
    gui_id: usize,
    obj_buf: *Wgpu.ObjectBuffer(Self),
) void {
    for (obj_buf.list.items, 0..) |h, i| {
        if (gui_id == h.grouping_id) {
            Wgpu.writeToObjectBuffer(gctx, Self, [4]f32, "color", .{
                .obj_buf = obj_buf.*,
                .index = i,
                .value = .{ 0, 0.5, 1, 0 },
            });
        }
    }
}

pub fn clearHover(
    gctx: *zgpu.GraphicsContext,
    obj_buf: *Wgpu.ObjectBuffer(Self),
) void {
    for (obj_buf.list.items, 0..) |_, i| {
        Wgpu.writeToObjectBuffer(gctx, Self, [4]f32, "color", .{
            .obj_buf = obj_buf.*,
            .index = i,
            .value = .{ 0, 0, 0, 0 },
        });
    }
}
