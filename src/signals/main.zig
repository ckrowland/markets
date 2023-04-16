const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const zm = @import("zmath");
const math = std.math;
const wgpu = zgpu.wgpu;
const Plot = @import("plot.zig");
const Parameter = @import("parameter.zig");
const Waves = @import("wave.zig");
const Wave = Waves.Wave;

const window_title = "FFT Demo";

const Self = @This();

depth_texture: zgpu.TextureHandle,
depth_texture_view: zgpu.TextureViewHandle,
random: Wave,
input: Wave,
output: Wave,
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, gctx: *zgpu.GraphicsContext) !Self {
    var randomWave = Waves.Wave.init(allocator, 0, .cos);
    randomWave.createWave();
    
    var inputWave = Waves.Wave.init(allocator, 1, .sin);
    inputWave.createComparisonWave(&randomWave);
    
    var outputWave = Waves.Wave.init(allocator, 2, .result);
    outputWave.multiplyWaves(&randomWave, &inputWave);

    // Create a depth texture and its 'view'.
    const depth = createDepthTexture(gctx);

    return Self{
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .random = randomWave,
        .input = inputWave,
        .output = outputWave,
    };
}

pub fn deinit(demo: *Self) void {
    demo.random.deinit();
    demo.input.deinit();
    demo.output.deinit();
    demo.* = undefined;
}

pub fn update(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    demo.input.createComparisonWave(&demo.random);
    demo.output.multiplyWaves(&demo.random, &demo.input);
    
    Parameter.window(demo, gctx);
    Plot.window(demo, gctx);
    // zgui.plot.showDemoWindow(null);
}

pub fn draw(demo: *Self, gctx: *zgpu.GraphicsContext) void {
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
