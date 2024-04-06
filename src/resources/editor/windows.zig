const std = @import("std");
const zgui = @import("zgui");
const zgpu = @import("zgpu");

pub const window_flags = .{
    .popen = null,
    .flags = zgui.WindowFlags.no_decoration,
};

pub const ParametersWindow = PercentArgs{
    .x = 0.0,
    .y = 0.0,
    .w = 0.25,
    .h = 0.75,
    .margin = 0.02,
};

pub const StatsWindow = PercentArgs{
    .x = 0.0,
    .y = 0.75,
    .w = 1.0,
    .h = 0.25,
    .margin = 0.02,
    .no_margin = .{ .top = true },
};

pub const PercentArgs = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    margin: f32,
    no_margin: struct {
        top: bool = false,
        bottom: bool = false,
        left: bool = false,
        right: bool = false,
    } = .{},
    // flags: zgui.WindowFlags = .{},
};

pub fn setNextWindowSize(
    gctx: *zgpu.GraphicsContext,
    p_width: f32,
    p_height: f32,
) void {
    std.debug.assert(0.0 <= p_width and p_width <= 1.0);
    std.debug.assert(0.0 <= p_height and p_height <= 1.0);
    const width = @as(f32, @floatFromInt(gctx.swapchain_descriptor.width));
    const height = @as(f32, @floatFromInt(gctx.swapchain_descriptor.height));
    zgui.setNextWindowSize(.{
        .w = width * p_width,
        .h = height * p_height,
    });
}

pub fn setNextWindow(gctx: *zgpu.GraphicsContext, args: PercentArgs) void {
    std.debug.assert(0.0 <= args.x and args.x <= 1.0);
    std.debug.assert(0.0 <= args.y and args.y <= 1.0);
    std.debug.assert(0.0 <= args.w and args.w <= 1.0);
    std.debug.assert(0.0 <= args.h and args.h <= 1.0);
    std.debug.assert(0.0 <= args.margin and args.margin <= 1.0);
    const width = @as(f32, @floatFromInt(gctx.swapchain_descriptor.width));
    const height = @as(f32, @floatFromInt(gctx.swapchain_descriptor.height));
    const margin_x = width * args.margin;
    const margin_y = height * args.margin;
    const margin_pixels = @min(margin_x, margin_y);
    var x = width * args.x + margin_pixels;
    var y = height * args.y + margin_pixels;
    var w = width * args.w - (margin_pixels * 2);
    var h = height * args.h - (margin_pixels * 2);

    if (args.no_margin.top) {
        y -= margin_pixels;
        h += margin_pixels;
    }
    if (args.no_margin.bottom) {
        h += margin_pixels;
    }
    if (args.no_margin.left) {
        x -= margin_pixels;
        w += margin_pixels;
    }
    if (args.no_margin.right) {
        w += margin_pixels;
    }

    zgui.setNextWindowPos(.{
        .x = x,
        .y = y,
    });

    zgui.setNextWindowSize(.{
        .w = w,
        .h = h,
    });
}
