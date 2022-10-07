const std = @import("std");
const zgpu = @import("zgpu");
const zm = @import("zmath");
const GraphicsContext = zgpu.GraphicsContext;
const wgpu = zgpu.wgpu;
const main = @import("bloodstream.zig");
const DemoState = main.DemoState;

pub fn getUniformBindGroupLayout(
    gctx: *GraphicsContext,
) zgpu.BindGroupLayoutHandle {
    const layout_entry = .{
        zgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
    };
    const uniform_bgl = gctx.createBindGroupLayout(&layout_entry);
    return uniform_bgl;
}

pub fn getUniformBindGroup(
    gctx: *GraphicsContext,
    bgl: zgpu.BindGroupLayoutHandle
) zgpu.BindGroupHandle {

    return gctx.createBindGroup(
        bgl,
        &.{
            .{
                .binding = 0,
                .buffer_handle = gctx.uniforms.buffer,
                .offset = 0,
                .size = @sizeOf(zm.Mat)
            },
        }
    );
}
