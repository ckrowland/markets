const std = @import("std");
const zemscripten = @import("zemscripten");
const slider = @import("main.zig");
pub const panic = zemscripten.panic;
pub const std_options = std.Options{
    .logFn = zemscripten.log,
};

export fn main() c_int {
    _ = zemscripten.setResizeCallback(&resizeCallback, false, &demo);
    zemscripten.setMainLoop(mainLoopCallback, null, false);
    return 0;
}

var initialized = false;
var demo: slider.DemoState = undefined;
var gpa = zemscripten.EmmalocAllocator{};
const allocator = gpa.allocator();

export fn mainLoopCallback() void {
    if (initialized == false) {
        demo = slider.init(allocator) catch |err| {
            std.log.err("slider.init failed with error: {s}", .{@errorName(err)});
            return;
        };
        initialized = true;

        var width: f64 = 0;
        var height: f64 = 0;
        const result = zemscripten.getElementCssSize("#canvas", &width, &height);
        if (result != .success) unreachable;
        demo.window.setSize(@intFromFloat(width), @intFromFloat(height));
    }
    slider.updateAndRender(&demo) catch |err| {
        std.log.err("sdl_demo.tick failed with error: {s}", .{@errorName(err)});
    };
}

pub fn resizeCallback(
    event_type: i16,
    event: *anyopaque,
    user_data: ?*anyopaque,
) callconv(.C) c_int {
    _ = event_type;
    _ = event;
    var width: f64 = 0;
    var height: f64 = 0;
    const slider_demo: *slider.DemoState = @ptrCast(@alignCast(user_data.?));
    const result = zemscripten.getElementCssSize("#canvas", &width, &height);
    if (result != .success) return 0;

    slider_demo.window.setSize(@intFromFloat(width), @intFromFloat(height));
    if (slider_demo.gctx.present() == .swap_chain_resized) {
        slider_demo.content_scale = slider.getContentScale(slider_demo.window);
        slider.setImguiContentScale(slider_demo.content_scale);
        slider.updateAspectRatio(slider_demo);
    }

    return 1;
}
