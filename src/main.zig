const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Random = @import("resources/random/main.zig");
const Editor = @import("resources/editor/main.zig");
const Signals = @import("signals/main.zig");
const Arithmetic = @import("arithmetic/main.zig");
const Window = @import("windows.zig");
const Camera = @import("camera.zig");

const content_dir = @import("build_options").content_dir;
const window_title = "Visual Simulations";

pub const DemoState = struct {
    allocator: std.mem.Allocator,
    gctx: *zgpu.GraphicsContext,
    current_demo: i32,
    aspect: f32,
    content_scale: [2]f32,
    demos: Demos = undefined,
};

const Demos = struct {
    random: Random,
    editor: Editor,
    signals: Signals,
    arithmetic: Arithmetic,
};

fn init(allocator: std.mem.Allocator, window: *zglfw.Window) !DemoState {
    const gctx = try zgpu.GraphicsContext.create(allocator, window, .{});
    return DemoState{
        .allocator = allocator,
        .gctx = gctx,
        .current_demo = 0,
        .aspect = Camera.getAspectRatio(gctx),
        .content_scale = gctx.window.getContentScale(),
    };
}

fn deinit(allocator: std.mem.Allocator, demo: *DemoState) void {
    demo.gctx.destroy(allocator);
    demo.demos.random.deinit();
    demo.demos.editor.deinit();
    demo.demos.signals.deinit();
    demo.demos.arithmetic.deinit();
}

fn update(demo: *DemoState) !void {
    const sd = demo.gctx.swapchain_descriptor;
    zgui.backend.newFrame(sd.width, sd.height);
    Window.commonGui(demo);
    switch (demo.current_demo) {
        0 => {
            demo.demos.random.update(demo.gctx);
        },
        1 => {
            try demo.demos.editor.update(demo.gctx);
        },
        2 => {
            demo.demos.signals.update(demo.gctx);
        },
        3 => {
            demo.demos.arithmetic.update(demo.gctx);
        },
        else => {},
    }
}

fn draw(demo: *DemoState) void {
    switch (demo.current_demo) {
        0 => {
            demo.demos.random.draw(demo.gctx);
        },
        1 => {
            demo.demos.editor.draw(demo.gctx);
        },
        2 => {
            demo.demos.signals.draw(demo.gctx);
        },
        3 => {
            demo.demos.arithmetic.draw(demo.gctx);
        },
        else => {},
    }

    if (demo.gctx.present() == .swap_chain_resized) {
        demo.aspect = Camera.getAspectRatio(demo.gctx);
        demo.content_scale = demo.gctx.window.getContentScale();
        setImguiContentScale(demo);

        demo.demos.editor.updateAspectRatio(demo.gctx);
        demo.demos.random.updateAspectRatio(demo.gctx);
        demo.demos.signals.updateDepthTexture(demo.gctx);
        demo.demos.arithmetic.updateDepthTexture(demo.gctx);
    }
}

fn setImguiContentScale(demo: *DemoState) void {
    const scale = @max(demo.content_scale[0], demo.content_scale[1]);
    zgui.getStyle().* = zgui.Style.init();
    zgui.getStyle().scaleAllSizes(scale);
}

pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    const window = zglfw.Window.create(725, 900, window_title, null) catch {
        std.log.err("Failed to create demo window.", .{});
        return;
    };
    defer window.destroy();
    window.setSizeLimits(400, 400, -1, -1);
    window.setPos(0, 0);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var demo = try init(allocator, window);
    demo.demos = .{
        .random = try Random.init(&demo),
        .editor = try Editor.init(&demo),
        .signals = try Signals.init(&demo),
        .arithmetic = try Arithmetic.init(&demo),
    };
    defer deinit(allocator, &demo);

    zgui.init(allocator);
    defer zgui.deinit();
    zgui.plot.init();
    defer zgui.plot.deinit();

    const scale = @max(demo.content_scale[0], demo.content_scale[1]);
    _ = zgui.io.addFontFromFile(
        content_dir ++ "/fonts/Roboto-Medium.ttf",
        20.0 * scale,
    );
    setImguiContentScale(&demo);

    zgui.backend.init(
        window,
        demo.gctx.device,
        @intFromEnum(zgpu.GraphicsContext.swapchain_format),
    );
    defer zgui.backend.deinit();
    zgui.io.setIniFilename(null);

    while (!window.shouldClose()) {
        zglfw.pollEvents();
        if (!window.getAttribute(.focused)) {
            continue;
        }
        try update(&demo);
        draw(&demo);
    }
}
