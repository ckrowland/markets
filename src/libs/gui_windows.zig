const std = @import("std");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");

pub const Pos = struct {
    x: f32,
    y: f32,
    margin: struct {
        percent: f32 = 0.02,
        top: bool = true,
        bottom: bool = true,
        left: bool = true,
        right: bool = true,
    } = .{},
};

pub fn setupWindowPos(sd: wgpu.SwapChainDescriptor, pos: Pos) void {
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
    const margin_x = width * pos.margin.percent;
    const margin_y = height * pos.margin.percent;
    const pos_margin_pixels = @min(margin_x, margin_y);

    var x = width * pos.x;
    if (pos.margin.left) {
        x += pos_margin_pixels;
    }

    var y = height * pos.y;
    if (pos.margin.top) {
        y += pos_margin_pixels;
    }

    zgui.setNextWindowPos(.{ .x = x, .y = y });
}

pub fn setupWindowSize(sd: wgpu.SwapChainDescriptor, size: Pos) void {
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
    const margin_x = width * size.margin.percent;
    const margin_y = height * size.margin.percent;
    const size_margin_pixels = @min(margin_x, margin_y);

    var w = width * size.x;
    if (size.margin.left) {
        w -= size_margin_pixels;
    }
    if (size.margin.right) {
        w -= size_margin_pixels;
    }

    var h = height * size.y;
    if (size.margin.top) {
        h -= size_margin_pixels;
    }
    if (size.margin.bottom) {
        h -= size_margin_pixels;
    }
    zgui.setNextWindowSize(.{ .w = w, .h = h });
}
