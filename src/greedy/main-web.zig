const std = @import("std");
const zemscripten = @import("zemscripten");
const variable = @import("main.zig");
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
var demo: variable.DemoState = undefined;
var gpa = zemscripten.EmmalocAllocator{};
const allocator = gpa.allocator();

export fn mainLoopCallback() void {
    if (initialized == false) {
        demo = variable.init(allocator) catch |err| {
            std.log.err("variable.init failed with error: {s}", .{@errorName(err)});
            return;
        };
        initialized = true;
    }
    variable.updateAndRender(&demo) catch |err| {
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
    const variable_demo: *variable.DemoState = @ptrCast(@alignCast(user_data.?));
    const result = zemscripten.getElementCssSize("#canvas", &width, &height);
    if (result != .success) return 0;

    variable_demo.window.setSize(@intFromFloat(width), @intFromFloat(height));
    if (variable_demo.gctx.present() == .swap_chain_resized) {
        variable_demo.content_scale = variable.getContentScale(variable_demo.window);
        variable.setImguiContentScale(variable_demo.content_scale);
        variable.updateAspectRatio(variable_demo);
    }

    return 1;
}
