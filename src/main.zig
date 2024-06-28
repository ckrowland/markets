const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zstbi = @import("zstbi");
const Random = @import("resources/random/main.zig");
const Editor = @import("resources/editor/main.zig");
const Variable = @import("resources/variable/main.zig");

const Selection = enum {
    Random,
    Editor,
    Variable,
};
pub var selection = Selection.Variable;
pub fn selectionGui() void {
    zgui.text("Pick a demo", .{});
    _ = zgui.comboFromEnum("##tab_bar", &selection);
    zgui.dummy(.{ .w = 1, .h = 10 });
}
pub var quit = false;

pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    // Change current working directory to where the executable is located.
    var buffer: [1024]u8 = undefined;
    const path = std.fs.selfExeDirPath(buffer[0..]) catch ".";
    std.posix.chdir(path) catch {};

    zglfw.windowHintTyped(.client_api, .no_api);

    const window = try zglfw.Window.create(1600, 1000, "Simulations", null);
    defer window.destroy();
    window.setSizeLimits(400, 400, -1, -1);
    window.setPos(0, 0);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
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

    const cs = window.getContentScale();
    const content_scale = @max(cs[0], cs[1]);
    zgui.io.setIniFilename(null);
    _ = zgui.io.addFontFromFile(
        "content/fonts/Roboto-Medium.ttf",
        22.0 * content_scale,
    );

    zgui.backend.init(
        window,
        gctx.device,
        @intFromEnum(zgpu.GraphicsContext.swapchain_format),
        @intFromEnum(wgpu.TextureFormat.undef),
    );
    defer zgui.backend.deinit();

    while (!quit) {
        switch (selection) {
            Selection.Random => {
                var demo = try Random.init(gctx, allocator, window);
                defer Random.deinit(&demo);
                quit = window.shouldClose();
                while (!quit) {
                    quit = window.shouldClose();
                    switch (selection) {
                        Selection.Variable, Selection.Editor => break,
                        Selection.Random => {
                            Random.update(&demo, &selectionGui);
                            Random.draw(&demo);
                        },
                    }
                }
            },
            Selection.Editor => {
                var demo = try Editor.init(gctx, allocator, window);
                defer Editor.deinit(&demo);
                quit = window.shouldClose();
                while (!quit) {
                    quit = window.shouldClose();
                    switch (selection) {
                        Selection.Random, Selection.Variable => break,
                        Selection.Editor => {
                            try Editor.update(&demo, &selectionGui);
                            Editor.draw(&demo);
                        },
                    }
                }
            },
            Selection.Variable => {
                var demo = try Variable.init(gctx, allocator, window);
                defer Variable.deinit(&demo);
                quit = window.shouldClose();
                while (!quit) {
                    quit = window.shouldClose();
                    switch (selection) {
                        Selection.Random, Selection.Editor => break,
                        Selection.Variable => {
                            Variable.update(&demo, &selectionGui);
                            Variable.draw(&demo);
                        },
                    }
                }
            },
        }
    }
}
