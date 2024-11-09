const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const zpool = @import("zpool");
const zmath = @import("zmath");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zstbi = @import("zstbi");
const Random = @import("random/main.zig");
const Editor = @import("editor/main.zig");
const Variable = @import("variable/main.zig");
const zemscripten = @import("zemscripten");
const emscripten = @import("builtin").target.os.tag == .emscripten;

pub const std_options = std.Options{
    .logFn = if (emscripten) zemscripten.log else std.log.defaultLog,
};

pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    // Change current working directory to where the executable is located.
    if (!emscripten) {
        var buffer: [1024]u8 = undefined;
        const path = std.fs.selfExeDirPath(buffer[0..]) catch ".";
        std.posix.chdir(path) catch {};
    }

    const window = try zglfw.Window.create(1600, 900, "Simulations", null);
    defer window.destroy();

    window.setSizeLimits(400, 400, -1, -1);
    if (!emscripten) window.setPos(50, 50);
    //zglfw.windowHintTyped(.client_api, .no_api);

    var gpa = blk: {
        if (emscripten) {
            break :blk zemscripten.EmmalocAllocator{};
        } else {
            break :blk std.heap.GeneralPurposeAllocator(.{}){};
        }
    };
    defer _ = if (!emscripten) gpa.deinit();

    const allocator = gpa.allocator();

    zstbi.init(allocator);
    defer zstbi.deinit();

    const gctx = try zgpu.GraphicsContext.create(
        allocator,
        .{
            .window = window,
            .fn_getTime = @ptrCast(&zglfw.getTime),
            .fn_getFramebufferSize = @ptrCast(&zglfw.Window.getFramebufferSize),
            .fn_getWin32Window = @ptrCast(&zglfw.getWin32Window),
            .fn_getX11Display = @ptrCast(&zglfw.getX11Display),
            .fn_getX11Window = @ptrCast(&zglfw.getX11Window),
            .fn_getWaylandDisplay = @ptrCast(&zglfw.getWaylandDisplay),
            .fn_getWaylandSurface = @ptrCast(&zglfw.getWaylandWindow),
            .fn_getCocoaWindow = @ptrCast(&zglfw.getCocoaWindow),
        },
        .{},
    );
    defer gctx.destroy(allocator);

    zgui.init(allocator);
    defer zgui.deinit();
    zgui.plot.init();
    defer zgui.plot.deinit();
    zgui.io.setIniFilename(null);

    const content_scale = getContentScale(window);
    zgui.getStyle().scaleAllSizes(content_scale);

    const font_size = switch (emscripten) {
        true => 28.0 * content_scale,
        false => 20.0 * content_scale,
    };
    _ = zgui.io.addFontFromFile("content/fonts/Roboto-Medium.ttf", font_size);

    zgui.backend.init(
        window,
        gctx.device,
        @intFromEnum(zgpu.GraphicsContext.swapchain_format),
        @intFromEnum(wgpu.TextureFormat.undef),
    );
    defer zgui.backend.deinit();

    var demo = try Random.init(gctx, allocator, window);
    defer Random.deinit(&demo);

    if (emscripten) {
        _ = zemscripten.setResizeCallback(&resizeCallback, false, &demo);
        zemscripten.requestAnimationFrameLoop(&tickEmcripten, &demo);
        while (true) zemscripten.emscripten_sleep(1000);
    } else while (!window.shouldClose()) {
        try tick(&demo);
    }
}

pub fn tick(demo: *Random.DemoState) !void {
    zglfw.pollEvents();
    zgui.backend.newFrame(
        demo.gctx.swapchain_descriptor.width,
        demo.gctx.swapchain_descriptor.height,
    );
    Random.update(demo);
    Random.draw(demo);
    demo.window.swapBuffers();
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
    const demo: *Random.DemoState = @ptrCast(@alignCast(user_data.?));
    const result = zemscripten.getElementCssSize("#canvas", &width, &height);
    if (result != .success) return 0;

    demo.window.setSize(@intFromFloat(width), @intFromFloat(height));
    if (demo.gctx.present() == .swap_chain_resized) {
        demo.content_scale = getContentScale(demo.window);
        setImguiContentScale(demo.content_scale);
        Random.updateAspectRatio(demo);
    }

    return 1;
}

fn setImguiContentScale(scale: f32) void {
    zgui.getStyle().* = zgui.Style.init();
    zgui.getStyle().scaleAllSizes(scale);
}

fn getContentScale(window: *zglfw.Window) f32 {
    const content_scale = window.getContentScale();
    return @max(1, @max(content_scale[0], content_scale[1]));
}

usingnamespace if (emscripten) struct {
    pub export fn tickEmcripten(time: f64, user_data: ?*anyopaque) callconv(.C) c_int {
        _ = time;
        const demo: *Random.DemoState = @ptrCast(@alignCast(user_data.?));
        if (demo.gctx.canRender()) tick(demo) catch |err| {
            std.log.err("animation frame canceled! tick failed with: {}", .{err});
            return 0; // FALSE - stop animation frame callback loop
        } else {
            std.log.warn("canRender(): Frame skipped!", .{});
        }
        return 1; // TRUE - continue animation frame callback loop
    }
} else struct {};
extern fn tickEmcripten(time: f64, user_data: ?*anyopaque) callconv(.C) c_int;
