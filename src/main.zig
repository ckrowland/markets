const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Resources = @import("resources/main.zig");
const Signals = @import("signals/main.zig");
const Window = @import("windows.zig");

const content_dir = @import("build_options").content_dir;
const window_title = "Visual Simulations";

pub const DemoState = struct {
    number: i32,
    gctx: *zgpu.GraphicsContext,
    resources: Resources,
    signals: Signals,
};

fn init(allocator: std.mem.Allocator, window: *zglfw.Window) !DemoState {
    const gctx = try zgpu.GraphicsContext.create(allocator, window);
    const signals = try Signals.init(allocator, gctx);
    const resources = try Resources.init(allocator, gctx);
    return DemoState{
        .number = 0,
        .gctx = gctx,
        .signals = signals,
        .resources = resources,
    };
}

fn deinit(allocator: std.mem.Allocator, demo: *DemoState) void {
    demo.gctx.destroy(allocator);
    demo.resources.deinit();
    demo.signals.deinit();
}

fn update(demo: *DemoState) void {
    const sd = demo.gctx.swapchain_descriptor;
    zgui.backend.newFrame(sd.width, sd.height);
    Window.commonGui(demo);
    switch (demo.number) {
        0 => {
            demo.resources.update(demo.gctx);
        },
        1 => {
            demo.signals.update(demo.gctx);
        },
        else => {},
    }
}

fn draw(demo: *DemoState) void {
    switch (demo.number) {
        0 => {
            demo.resources.draw(demo.gctx);
        },
        1 => {
            demo.signals.draw(demo.gctx);
        },
        else => {
            demo.resources.draw(demo.gctx);
        },
    }
}

pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    //zglfw.Hint.reset();
    //zglfw.Hint.set(.cocoa_retina_framebuffer, 1);
    //zglfw.Hint.set(.client_api, 0);
    const window = zglfw.Window.create(1600, 1000, window_title, null) catch {
        std.log.err("Failed to create demo window.", .{});
        return;
    };
    defer window.destroy();
    window.setSizeLimits(400, 400, -1, -1);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var demo = try init(allocator, window);
    defer deinit(allocator, &demo);

    const scale_factor = scale_factor: {
        const scale = window.getContentScale();
        break :scale_factor math.max(scale[0], scale[1]);
    };

    zgui.init(allocator);
    defer zgui.deinit();

    zgui.plot.init();
    defer zgui.plot.deinit();

    _ = zgui.io.addFontFromFile(content_dir ++ "Roboto-Medium.ttf", 19.0 * scale_factor);

    zgui.backend.init(
        window,
        demo.gctx.device,
        @enumToInt(zgpu.GraphicsContext.swapchain_format),
    );
    defer zgui.backend.deinit();

    zgui.getStyle().scaleAllSizes(scale_factor);

    while (!window.shouldClose()) {
        zglfw.pollEvents();
        if (!window.getAttribute(.focused)) {
            continue;
        }
        update(&demo);
        draw(&demo);
    }
}
