const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Wgpu = @import("../resources/wgpu.zig");
const Main = @import("../../main.zig");
const Camera = @import("../camera.zig");
const Window = @import("../windows.zig");
const Config = @import("config.zig");

const content_dir = @import("build_options").content_dir;

pub const Parameters = struct {
    aspect: f32,
};

const Self = @This();

block_rp: zgpu.RenderPipelineHandle,
depth_texture: zgpu.TextureHandle,
depth_texture_view: zgpu.TextureViewHandle,
params: Parameters,
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, gctx: *zgpu.GraphicsContext) !Self {
    const aspect = Camera.getAspectRatio(gctx);
    const params = Parameters{ .aspect = aspect };
    const depth = Wgpu.createDepthTexture(gctx);

    return Self{
        .block_rp = Wgpu.createRenderPipeline(
            gctx,
            Config.block_render_pipeline,
        ),
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .params = params,
    };
}

pub fn deinit(demo: *Self) void {
    demo.* = undefined;
}

pub fn update(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    _ = demo;
    Window.setNextWindow(gctx, Window.ParametersWindow);
    if (zgui.begin("Parameters", Window.window_flags)) {
        zgui.pushIntId(2);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        zgui.text("Number Of Producers", .{});
        zgui.popId();
    }
    zgui.end();
}

pub fn draw(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    //const cam_world_to_clip = Camera.getObjectToClipMat(gctx);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        // Draw ImGui
        {
            const pass = zgpu.beginRenderPassSimple(
                encoder,
                .load,
                back_buffer_view,
                null,
                null,
                null,
            );
            defer zgpu.endReleasePass(pass);
            zgui.backend.draw(pass);
        }

        break :commands encoder.finish(null);
    };
    defer commands.release();

    gctx.submit(&.{commands});

    if (gctx.present() == .swap_chain_resized) {
        demo.updateAspectRatio(gctx);
    }
}

pub fn updateDepthTexture(state: *Self, gctx: *zgpu.GraphicsContext) void {
    // Release old depth texture.
    gctx.releaseResource(state.depth_texture_view);
    gctx.destroyResource(state.depth_texture);

    // Create a new depth texture to match the new window size.
    const depth = Wgpu.createDepthTexture(gctx);
    state.depth_texture = depth.texture;
    state.depth_texture_view = depth.view;
}

pub fn updateAspectRatio(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    demo.updateDepthTexture(gctx);

    // Update grid positions to new aspect ratio
    const aspect = Camera.getAspectRatio(gctx);
    demo.params.aspect = aspect;
}
