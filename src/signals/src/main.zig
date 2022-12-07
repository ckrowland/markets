const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Window = @import("windows.zig");
const Plot = @import("plot.zig");
const Parameter = @import("parameter.zig");
const Waves = @import("wave.zig");

pub const Parameters = struct {
    shift: f32 = 0,
    num_points_per_cycle: u32 = 10,
    num_cycles: u32 = 1,
};

pub const DemoState = struct {
    gctx: *zgpu.GraphicsContext,
    depth_texture: zgpu.TextureHandle,
    depth_texture_view: zgpu.TextureViewHandle,
    params: Parameters,
    allocator: std.mem.Allocator,
};

fn init(allocator: std.mem.Allocator, window: zglfw.Window) !DemoState {
    const gctx = try zgpu.GraphicsContext.create(allocator, window);

    // Create a depth texture and its 'view'.
    const depth = createDepthTexture(gctx);

    return DemoState{
        .gctx = gctx,
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .params = Parameters{},
    };
}

fn deinit(allocator: std.mem.Allocator, demo: *DemoState) void {
    demo.gctx.destroy(allocator);
    demo.* = undefined;
}

fn update(demo: *DemoState) void {
    zgui.backend.newFrame(demo.gctx.swapchain_descriptor.width, demo.gctx.swapchain_descriptor.height);

    if (Window.beginGuiWindow(demo, Window.Parameters)) {
        defer zgui.end();
        Parameter.setup();
        Parameter.displayFPS(demo);
        Parameter.slider(
            f32,
            &demo.params.shift,
            Parameter.Shift,
        );
        Parameter.slider(
            u32,
            &demo.params.num_points_per_cycle,
            Parameter.NumPointsPerCycle,
        );
        Parameter.slider(
            u32,
            &demo.params.num_cycles,
            Parameter.NumCycles,
        );
    }

    if (Window.beginGuiWindow(demo, Window.Plots)) {
        defer zgui.end();

        if (zgui.plot.beginPlot("##first", .{})) {
            defer zgui.plot.endPlot();
            Plot.setup();

            var sin_wave = Waves.Wave.createSinWave(demo);
            defer sin_wave.deinit();

            zgui.plot.plotLine("Sin", f32, .{
                .xv = sin_wave.xv.items,
                .yv = sin_wave.yv.items,
            });
        }
    }
}

fn draw(demo: *DemoState) void {
    const gctx = demo.gctx;

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();
        {
            const pass = zgpu.beginRenderPassSimple(encoder, .load, back_buffer_view, null, null, null);
            defer zgpu.endReleasePass(pass);
            zgui.backend.draw(pass);
        }

        break :commands encoder.finish(null);
    };
    defer commands.release();

    gctx.submit(&.{commands});

    if (gctx.present() == .swap_chain_resized) {
        // Release old depth texture.
        gctx.releaseResource(demo.depth_texture_view);
        gctx.destroyResource(demo.depth_texture);

        // Create a new depth texture to match the new window size.
        const depth = createDepthTexture(gctx);
        demo.depth_texture = depth.texture;
        demo.depth_texture_view = depth.view;
    }
}

fn createDepthTexture(gctx: *zgpu.GraphicsContext) struct {
    texture: zgpu.TextureHandle,
    view: zgpu.TextureViewHandle,
} {
    const texture = gctx.createTexture(.{
        .usage = .{ .render_attachment = true },
        .dimension = .tdim_2d,
        .size = .{
            .width = gctx.swapchain_descriptor.width,
            .height = gctx.swapchain_descriptor.height,
            .depth_or_array_layers = 1,
        },
        .format = .depth32_float,
        .mip_level_count = 1,
        .sample_count = 1,
    });
    const view = gctx.createTextureView(texture, .{});
    return .{ .texture = texture, .view = view };
}

pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    zglfw.defaultWindowHints();
    zglfw.windowHint(.cocoa_retina_framebuffer, 1);
    zglfw.windowHint(.client_api, 0);
    const window = try zglfw.createWindow(1600, 1000, "Signal Sandbox", null, null);
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

    const content_dir = @import("build_options").content_dir;
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
        update(&demo);
        draw(&demo);
    }
}
