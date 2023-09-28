const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Wgpu = @import("../resources/wgpu.zig");
const Main = @import("../main.zig");
const Camera = @import("camera.zig");
const Window = @import("../windows.zig");
const Config = @import("config.zig");
const Block = @import("block.zig");
const content_dir = @import("build_options").content_dir;

const Self = @This();

allocator: std.mem.Allocator,
aspect: f32,
block: Block.GraphicsObjects,
render_bg: zgpu.BindGroupHandle,
depth: Wgpu.Depth,

pub fn init(demo: *Main.DemoState) !Self {
    const gctx = demo.gctx;
    return Self{
        .allocator = demo.allocator,
        .aspect = Camera.getAspectRatio(gctx),
        .block = Block.init(gctx),
        .render_bg = Wgpu.createUniformBindGroup(gctx),
        .depth = Wgpu.createDepthTexture(gctx),
    };
}

pub fn deinit(demo: *Self) void {
    demo.* = undefined;
}

pub fn update(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    _ = demo;
    _ = gctx;
}

pub fn draw(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    const cam_world_to_clip = Camera.getObjectToClipMat(gctx);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        pass: {
            const render_bind_group = gctx.lookupResource(demo.render_bg) orelse break :pass;
            const depth_view = gctx.lookupResource(demo.depth.view) orelse break :pass;
            const color_attachments = [_]wgpu.RenderPassColorAttachment{.{
                .view = back_buffer_view,
                .load_op = .clear,
                .store_op = .store,
            }};
            const depth_attachment = wgpu.RenderPassDepthStencilAttachment{
                .view = depth_view,
                .depth_load_op = .clear,
                .depth_store_op = .store,
                .depth_clear_value = 1.0,
            };
            const render_pass_info = wgpu.RenderPassDescriptor{
                .color_attachment_count = color_attachments.len,
                .color_attachments = &color_attachments,
                .depth_stencil_attachment = &depth_attachment,
            };

            const pass = encoder.beginRenderPass(render_pass_info);
            defer {
                pass.end();
                pass.release();
            }

            var mem = gctx.uniformsAllocate(zm.Mat, 1);
            mem.slice[0] = cam_world_to_clip;
            pass.setBindGroup(0, render_bind_group, &.{mem.offset});
            Wgpu.draw(demo.block.block, gctx, pass);
            // Wgpu.draw(demo.block.border, gctx, pass);
        }

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
}

pub fn updateDepthTexture(state: *Self, gctx: *zgpu.GraphicsContext) void {
    // Release old depth texture.
    gctx.releaseResource(state.depth.view);
    gctx.destroyResource(state.depth.texture);

    // Create a new depth texture to match the new window size.
    state.depth = Wgpu.createDepthTexture(gctx);
}
