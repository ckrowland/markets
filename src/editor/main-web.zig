const std = @import("std");
const zemscripten = @import("zemscripten");
const zglfw = @import("zglfw");
const editor = @import("main.zig");
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
var demo: editor.DemoState = undefined;
var gpa = zemscripten.EmmalocAllocator{};
const allocator = gpa.allocator();

export fn mainLoopCallback() void {
    if (initialized == false) {
        demo = editor.init(allocator) catch |err| {
            std.log.err("editor.init failed with error: {s}", .{@errorName(err)});
            return;
        };
        var width: f64 = 0;
        var height: f64 = 0;
        const result = zemscripten.getElementCssSize("#canvas", &width, &height);
        if (result != .success) unreachable;
        zglfw.setSize(demo.window, @intFromFloat(width), @intFromFloat(height));

        initialized = true;
        std.log.err("initialized", .{});
    }
    editor.updateAndRender(&demo) catch |err| {
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
    const editor_demo: *editor.DemoState = @ptrCast(@alignCast(user_data.?));
    const result = zemscripten.getElementCssSize("#canvas", &width, &height);
    if (result != .success) return 0;

    zglfw.setSize(editor_demo.window, @intFromFloat(width), @intFromFloat(height));
    if (editor_demo.gctx.present() == .swap_chain_resized) {
        editor_demo.content_scale = editor.getContentScale(editor_demo.window);
        editor.setImguiContentScale(editor_demo.content_scale);
        editor.updateAspectRatio(editor_demo);
    }

    return 1;
}
