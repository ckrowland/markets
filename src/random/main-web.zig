const std = @import("std");
const zemscripten = @import("zemscripten");
const random = @import("main.zig");
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
var demo: random.DemoState = undefined;
var gpa = zemscripten.EmmalocAllocator{};
const allocator = gpa.allocator();

export fn mainLoopCallback() void {
    if (initialized == false) {
        demo = random.init(allocator) catch |err| {
            std.log.err("random.init failed with error: {s}", .{@errorName(err)});
            return;
        };
        initialized = true;
    }
    random.updateAndRender(&demo) catch |err| {
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
    const random_demo: *random.DemoState = @ptrCast(@alignCast(user_data.?));
    const result = zemscripten.getElementCssSize("#canvas", &width, &height);
    if (result != .success) return 0;

    random_demo.window.setSize(@intFromFloat(width), @intFromFloat(height));
    if (random_demo.gctx.present() == .swap_chain_resized) {
        random_demo.content_scale = random.getContentScale(random_demo.window);
        random.setImguiContentScale(random_demo.content_scale);
        random.updateAspectRatio(random_demo);
    }

    return 1;
}
