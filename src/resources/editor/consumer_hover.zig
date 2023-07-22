const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Demo = @import("main.zig");
const Wgpu = @import("../wgpu.zig");
const Camera = @import("../../camera.zig");
const Mouse = @import("mouse.zig");
const Popups = @import("popups.zig");

const Self = @This();

absolute_home: [4]i32,
home: [4]f32,
color: [4]f32 = .{ 0, 0, 0, 0 },
radius: f32 = 60.0,
grouping_id: u32 = 0,

pub const z_pos = 0;

pub const Args = struct {
    absolute_home: [2]i32,
    home: [2]f32,
    grouping_id: u32,
};
pub fn create(args: Args) Self {
    return Self{
        .absolute_home = .{ args.absolute_home[0], args.absolute_home[1], z_pos, 1 },
        .home = .{ args.home[0], args.home[1], z_pos, 1 },
        .grouping_id = args.grouping_id,
    };
}

pub const hoverArgs = struct {
    consumer_hover: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
};
pub fn highlightConsumers(gctx: *zgpu.GraphicsContext, gui_id: usize, args: hoverArgs) void {
    Wgpu.setGroup(gctx, Self, .{
        .grouping_id = @intCast(u32, gui_id),
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
    
pub fn displayOnHover(gctx: *zgpu.GraphicsContext, args: hoverArgs) void {
    var consumer_hover = Wgpu.getAll(gctx, Self, .{
        .structs = args.consumer_hover,
        .num_structs = Wgpu.getNumStructs(gctx, Self, args.stats),
    }) catch return;

    for (consumer_hover) |circle| {
        const center = Camera.getPixelPosition(gctx, circle.absolute_home[0..2].*);
        const in_x = (center[0] - circle.radius) < args.mouse.cursor_pos[0] and
            args.mouse.cursor_pos[0] < (center[0] + circle.radius);
        const in_y = (center[1] - circle.radius) < args.mouse.cursor_pos[1] and
            args.mouse.cursor_pos[1] < (center[1] + circle.radius);

        var grouping_gui_open = false;
        if (circle.grouping_id < args.popups.popups.items.len) {
            grouping_gui_open = args.popups.popups.items[circle.grouping_id].open;
        }
        if (in_x and in_y or grouping_gui_open) {
            Wgpu.setGroup(gctx, Self, .{
                .grouping_id = circle.grouping_id,
                .setArgs = .{
                    .agents = args.consumer_hover,
                    .stats = args.stats,
                    .parameter = .{
                        .color = .{ 0, 0.5, 1, 0 },
                    },
                },
            });
            break;
        }
        
        Wgpu.setAll(gctx, Self, .{
            .agents = args.consumer_hover,
            .stats = args.stats,
            .parameter = .{
                .color = .{ 0, 0, 0, 0 },
            },
        });
    }
}
