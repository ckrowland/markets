const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const zm = @import("zmath");
const math = std.math;
const wgpu = zgpu.wgpu;
const Window = @import("windows.zig");
const Plot = @import("plot.zig");
const Parameter = @import("parameter.zig");
const Waves = @import("wave.zig");

const window_title = "FFT Demo";

pub const Parameters = struct {
    shift: u32 = 0,
    num_points_per_cycle: u32 = 10,
    num_cycles: u32 = 1,
};

pub const Signal = struct {
    params: Parameters = Parameters{},
    wave: Waves.Wave = undefined,
};

pub const DemoState = struct {
    gctx: *zgpu.GraphicsContext,
    depth_texture: zgpu.TextureHandle,
    depth_texture_view: zgpu.TextureViewHandle,
    input_one: Signal,
    input_two: Signal,
    output: Signal,
    allocator: std.mem.Allocator,
};

fn init(allocator: std.mem.Allocator, window: *zglfw.Window) !DemoState {
    const gctx = try zgpu.GraphicsContext.create(allocator, window);
    const params = Parameters{};

    // Create a depth texture and its 'view'.
    const depth = createDepthTexture(gctx);

    return DemoState{
        .gctx = gctx,
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .input_one = Signal{
            .params = params,
            .wave = Waves.Wave.initAndGenerate(allocator, params),
        },
        .input_two = Signal{
            .params = params,
            .wave = Waves.Wave.initAndGenerate(allocator, params),
        },
        .output = Signal{
            .params = params,
            .wave = Waves.Wave.init(allocator),
        },
    };
}

fn deinit(allocator: std.mem.Allocator, demo: *DemoState) void {
    demo.gctx.destroy(allocator);
    demo.input_one.wave.deinit();
    demo.input_two.wave.deinit();
    demo.output.wave.deinit();
    demo.* = undefined;
}

fn update(demo: *DemoState) void {
    zgui.backend.newFrame(demo.gctx.swapchain_descriptor.width, demo.gctx.swapchain_descriptor.height);

    Window.parameters(demo);
    Window.plots(demo);
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

    //zglfw.Window.Hint.reset();
    //zglfw.Window.Hint.set(.cocoa_retina_framebuffer, 1);
    //zglfw.Window.Hint.set(.client_api, 0);
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
