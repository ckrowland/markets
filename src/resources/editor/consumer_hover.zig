const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Demo = @import("main.zig");
const Wgpu = @import("../wgpu.zig");
const Camera = @import("../../camera.zig");
const Mouse = @import("mouse.zig");
const Popups = @import("popups.zig");
const Statistics = @import("../statistics.zig");
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
    hover_args: Consumer.Args,
    hover_buf: zgpu.BufferHandle,
    stat_obj: Wgpu.ObjectBuffer,
};
pub fn createAndAppend(gctx: *zgpu.GraphicsContext, args: AppendArgs) void {
    const num_structs = Wgpu.getNumStructs(gctx, Self, args.stat_obj);
    var hovers: [1]Self = .{
        create(args.hover_args),
    };
    Wgpu.appendBuffer(gctx, Self, .{
        .num_old_structs = num_structs,
        .buf = args.hover_buf,
        .structs = hovers[0..],
    });
    Statistics.setNumConsumerHovers(gctx, args.stat_obj, num_structs + 1);
}

pub const hoverArgs = struct {
    consumer_hover: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
};
pub fn highlightConsumers(gctx: *zgpu.GraphicsContext, gui_id: usize, args: hoverArgs) void {
    Wgpu.setGroup(gctx, Self, .{
        .grouping_id = @as(u32, @intCast(gui_id)),
        .setArgs = .{
            .agents = args.consumer_hover,
            .stats = args.stats,
            .parameter = .{
                .color = .{ 0, 0.5, 1, 0 },
            },
        },
    });
}

pub fn clearHover(gctx: *zgpu.GraphicsContext, args: hoverArgs) void {
    Wgpu.setAll(gctx, Self, .{
        .agents = args.consumer_hover,
        .stats = args.stats,
        .parameter = .{
            .color = .{ 0, 0, 0, 0 },
        },
    });
}
