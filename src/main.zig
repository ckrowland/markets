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
const Window = @import("windows.zig");

const content_dir = @import("build_options").content_dir;
const window_title = "Visual Simulations";

pub const DemoState = struct {
    number: i32,
    gctx: *zgpu.GraphicsContext,
    random: Random,
    editor: Editor,
    // signals: Signals,
};

fn init(allocator: std.mem.Allocator, window: *zglfw.Window) !DemoState {
    const gctx = try zgpu.GraphicsContext.create(allocator, window, .{});
    // const signals = try Signals.init(allocator, gctx);
    const random = try Random.init(allocator, gctx);
    const editor = try Editor.init(allocator, gctx);
    return DemoState{
        .number = 1,
        .gctx = gctx,
        .random = random,
        .editor = editor,
        // .signals = signals,
    };
}

fn deinit(allocator: std.mem.Allocator, demo: *DemoState) void {
    demo.gctx.destroy(allocator);
    demo.random.deinit();
    demo.editor.deinit();
    // demo.signals.deinit();
}

fn update(demo: *DemoState) !void {
    const sd = demo.gctx.swapchain_descriptor;
    zgui.backend.newFrame(sd.width, sd.height);
    Window.commonGui(demo);
    switch (demo.number) {
        0 => {
            demo.random.update(demo.gctx);
        },
        1 => {
            try demo.editor.update(demo.gctx);
        },
        // 2 => {
        //     demo.signals.update(demo.gctx);
        // },
        else => {},
    }
}

fn draw(demo: *DemoState) void {
    switch (demo.number) {
        0 => {
            demo.random.draw(demo.gctx);
        },
        1 => {
            demo.editor.draw(demo.gctx);
        },
        // 2 => {
        //     demo.signals.draw(demo.gctx);
        // },
        else => {
            demo.random.draw(demo.gctx);
        },
    }
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
    defer deinit(allocator, &demo);

    const scale_factor = scale_factor: {
        const scale = window.getContentScale();
        break :scale_factor @max(scale[0], scale[1]);
    };

    zgui.init(allocator);
    defer zgui.deinit();
    zgui.plot.init();
    defer zgui.plot.deinit();

    _ = zgui.io.addFontFromFile(content_dir ++ "/fonts/Roboto-Medium.ttf", 19.0 * scale_factor);

    zgui.backend.init(
        window,
        demo.gctx.device,
        @intFromEnum(zgpu.GraphicsContext.swapchain_format),
    );
    defer zgui.backend.deinit();

    zgui.getStyle().scaleAllSizes(scale_factor);
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
