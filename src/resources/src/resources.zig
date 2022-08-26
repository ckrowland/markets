const std = @import("std");
const math = std.math;
const glfw = @import("glfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = zgpu.zgui;
const zm = @import("zmath");
const array = std.ArrayList;
const random = std.crypto.random;
const F32x4 = @Vector(4, f32);
const Simulation = @import("simulation.zig");
const Shapes = @import("shapes.zig");
const wgsl = @import("shaders.zig");
const gui = @import("gui.zig");

const content_dir = @import("build_options").content_dir;
const window_title = "Resource Simulation";

pub const StagingBuffer = struct {
    slice: ?[]const f32 = null,
    buffer: wgpu.Buffer = undefined,
};

pub const Vertex = struct {
    position: [3]f32,
    color: [3]f32,
};

pub const DemoState = struct {
    gctx: *zgpu.GraphicsContext,

    producer_pipeline: zgpu.RenderPipelineHandle,
    consumer_pipeline: zgpu.RenderPipelineHandle,
    consumer_compute_pipeline: zgpu.ComputePipelineHandle,
    producer_compute_pipeline: zgpu.ComputePipelineHandle,
    bind_group: zgpu.BindGroupHandle,

    producer_vertex_buffer: zgpu.BufferHandle,
    producer_buffer: zgpu.BufferHandle,
    consumer_vertex_buffer: zgpu.BufferHandle,
    consumer_index_buffer: zgpu.BufferHandle,
    consumer_buffer: zgpu.BufferHandle,
    consumer_bind_group: zgpu.BindGroupHandle,
    num_transactions_buffer: zgpu.BufferHandle,
    num_transactions_buffer_copy: zgpu.BufferHandle,
    num_transactions: StagingBuffer,

    depth_texture: zgpu.TextureHandle,
    depth_texture_view: zgpu.TextureViewHandle,

    sim: Simulation,
    allocator: std.mem.Allocator,
};

fn init(allocator: std.mem.Allocator, window: glfw.Window) !DemoState {
    const gctx = try zgpu.GraphicsContext.init(allocator, window);

    // Render Pipeline and Bind Group
    const bind_group_layout = gctx.createBindGroupLayout(&.{
        zgpu.bglBuffer(0, .{ .vertex = true }, .uniform, true, 0),
    });
    defer gctx.releaseResource(bind_group_layout);
    const bind_group = gctx.createBindGroup(bind_group_layout,
                                            &[_]zgpu.BindGroupEntryInfo{
        .{ .binding = 0,
           .buffer_handle = gctx.uniforms.buffer,
           .offset = 0,
           .size = @sizeOf(zm.Mat) },
    });
    const pipeline_layout = gctx.createPipelineLayout(&.{bind_group_layout});
    defer gctx.releaseResource(pipeline_layout);
    const producer_pipeline = Shapes.createProducerPipeline(gctx, pipeline_layout);
    const consumer_pipeline = Shapes.createConsumerPipeline(gctx, pipeline_layout);


    // Simulation struct
    var sim = Simulation.init(allocator);
    sim.createAgents();

    // Create Compute Bind Group and Pipeline
    const compute_bgl = gctx.createBindGroupLayout(&.{
        zgpu.bglBuffer(0, .{ .compute = true }, .storage, true, 0),
        zgpu.bglBuffer(1, .{ .compute = true }, .storage, true, 0),
        zgpu.bglBuffer(2, .{ .compute = true }, .storage, true, 0),
    });
    defer gctx.releaseResource(compute_bgl);
    const compute_pl = gctx.createPipelineLayout(&.{compute_bgl});
    defer gctx.releaseResource(compute_pl);
    const consumer_compute_pipeline = Shapes.createConsumerComputePipeline(gctx, compute_pl);
    const producer_compute_pipeline = Shapes.createProducerComputePipeline(gctx, compute_pl);

    // Create Buffers
    const producer_buffer = Shapes.createProducerBuffer(gctx, sim.producers);
    const producer_vertex_buffer = Shapes.createProducerVertexBuffer(gctx, sim.params.producer_width);

    const num_vertices = 20;
    const consumer_vertex_buffer = Shapes.createConsumerVertexBuffer(gctx,
                                    sim.params.consumer_radius,
                                    num_vertices);
    const consumer_index_buffer = Shapes.createConsumerIndexBuffer(gctx, num_vertices);
    var consumer_buffer = Shapes.createConsumerBuffer(gctx, sim.consumers);

    const num_transactions_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .storage = true },
        .size = @sizeOf(f32),
    });
    const num_transactions_buffer_copy = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = @sizeOf(f32),
    });

    const num_transactions_data = [_]f32{ 0.0 };
    gctx.queue.writeBuffer(gctx.lookupResource(num_transactions_buffer).?, 0, f32, num_transactions_data[0..]);

    var num_transactions: StagingBuffer = .{
        .slice = null,
        .buffer = gctx.lookupResource(num_transactions_buffer_copy).?,
    };

    var consumer_bind_group = Shapes.createBindGroup(gctx,
                                                     sim,
                                                     compute_bgl,
                                                     consumer_buffer,
                                                     producer_buffer,
                                                     num_transactions_buffer);
 

    // Create a depth texture and its 'view'.
    const depth = createDepthTexture(gctx);

    return DemoState{
        .gctx = gctx,
        .producer_pipeline = producer_pipeline,
        .consumer_pipeline = consumer_pipeline,
        .consumer_compute_pipeline = consumer_compute_pipeline,
        .producer_compute_pipeline = producer_compute_pipeline,
        .bind_group = bind_group,
        .producer_vertex_buffer = producer_vertex_buffer,
        .producer_buffer = producer_buffer,
        .consumer_vertex_buffer = consumer_vertex_buffer,
        .consumer_index_buffer = consumer_index_buffer,
        .consumer_buffer = consumer_buffer,
        .consumer_bind_group = consumer_bind_group,
        .num_transactions_buffer = num_transactions_buffer,
        .num_transactions_buffer_copy = num_transactions_buffer_copy,
        .num_transactions = num_transactions,
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .sim = sim,
    };
}

fn deinit(allocator: std.mem.Allocator, demo: *DemoState) void {
    demo.gctx.deinit(allocator);
    demo.sim.deinit();
    demo.* = undefined;
}

fn update(demo: *DemoState) void {
    gui.update(demo);
}

fn draw(demo: *DemoState) void {
    const gctx = demo.gctx;
    const fb_width = gctx.swapchain_descriptor.width;
    const fb_height = gctx.swapchain_descriptor.height;
    //const t = @floatCast(f32, gctx.stats.time);
    //const frame_num = gctx.stats.gpu_frame_number;

    const cam_world_to_view = zm.lookAtLh(
        zm.f32x4(0.0, 0.0, -3000.0, 1.0),
        zm.f32x4(0.0, 0.0, 0.0, 1.0),
        zm.f32x4(0.0, 1.0, 0.0, 0.0),
    );
    const cam_view_to_clip = zm.perspectiveFovLh(
        0.25 * math.pi,
        @intToFloat(f32, fb_width) / @intToFloat(f32, fb_height),
        0.01,
        3001.0,
    );
    const cam_world_to_clip = zm.mul(cam_world_to_view, cam_view_to_clip);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        pass: {
            const cp = gctx.lookupResource(demo.producer_compute_pipeline) orelse break :pass;
            const ccp = gctx.lookupResource(demo.consumer_compute_pipeline) orelse break :pass;
            const bg = gctx.lookupResource(demo.consumer_bind_group) orelse break :pass;
            const bg_info = gctx.lookupResourceInfo(demo.consumer_bind_group) orelse break :pass;
            const first_offset = @intCast(u32, bg_info.entries[0].offset);
            const second_offset = @intCast(u32, bg_info.entries[1].offset);
            const third_offset = @intCast(u32, bg_info.entries[2].offset);
            const dynamic_offsets = &.{ first_offset, second_offset, third_offset};

            const pass = encoder.beginComputePass(null);
            defer {
                pass.end();
                pass.release();
            }
            pass.setPipeline(cp);
            pass.setBindGroup(0, bg, dynamic_offsets);
            const num_producers = @intToFloat(f32, demo.sim.producers.items.len);
            var workgroup_size = @floatToInt(u32, @ceil(num_producers / 64));
            pass.dispatchWorkgroups(workgroup_size, 1, 1);

            pass.setPipeline(ccp);
            const num_consumers = @intToFloat(f32, demo.sim.consumers.items.len);
            workgroup_size = @floatToInt(u32, @ceil(num_consumers / 64));
            pass.dispatchWorkgroups(workgroup_size, 1, 1);
        }

        // Copy transactions number to mapped buffer
        pass: {
            const buf = gctx.lookupResource(demo.num_transactions_buffer) orelse break :pass;
            const cp = gctx.lookupResource(demo.num_transactions_buffer_copy) orelse break :pass;
            encoder.copyBufferToBuffer(buf, 0, cp, 0, @sizeOf(f32));
        }

        pass: {
            const vb_info = gctx.lookupResourceInfo(demo.producer_vertex_buffer) orelse break :pass;
            const vpb_info = gctx.lookupResourceInfo(demo.producer_buffer) orelse break :pass;
            const cvb_info = gctx.lookupResourceInfo(demo.consumer_vertex_buffer) orelse break :pass;
            const cpb_info = gctx.lookupResourceInfo(demo.consumer_buffer) orelse break :pass;
            const cib_info = gctx.lookupResourceInfo(demo.consumer_index_buffer) orelse break :pass;
            const producer_pipeline = gctx.lookupResource(demo.producer_pipeline) orelse break :pass;
            const consumer_pipeline = gctx.lookupResource(demo.consumer_pipeline) orelse break :pass;
            const bind_group = gctx.lookupResource(demo.bind_group) orelse break :pass;
            const depth_view = gctx.lookupResource(demo.depth_texture_view) orelse break :pass;

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
            mem.slice[0] = zm.transpose(cam_world_to_clip);
            pass.setBindGroup(0, bind_group, &.{mem.offset});

            pass.setVertexBuffer(0, vb_info.gpuobj.?, 0, vb_info.size);
            pass.setVertexBuffer(1, vpb_info.gpuobj.?, 0, vpb_info.size);
            const num_producers = @intCast(u32, demo.sim.producers.items.len);
            pass.setPipeline(producer_pipeline);
            pass.draw(6, num_producers, 0, 0);

            pass.setVertexBuffer(0, cvb_info.gpuobj.?, 0, cvb_info.size);
            pass.setVertexBuffer(1, cpb_info.gpuobj.?, 0, cpb_info.size);
            pass.setIndexBuffer(cib_info.gpuobj.?, .uint32, 0, cib_info.size);
            const num_consumers = @intCast(u32, demo.sim.consumers.items.len);
            pass.setPipeline(consumer_pipeline);
            pass.drawIndexed(57, num_consumers, 0, 0, 0);


        }

        {
            const color_attachments = [_]wgpu.RenderPassColorAttachment{.{
                .view = back_buffer_view,
                .load_op = .load,
                .store_op = .store,
            }};
            const render_pass_info = wgpu.RenderPassDescriptor{
                .color_attachment_count = color_attachments.len,
                .color_attachments = &color_attachments,
            };
            const pass = encoder.beginRenderPass(render_pass_info);
            defer {
                pass.end();
                pass.release();
            }

            zgpu.gui.draw(pass);
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

pub fn buffersMappedCallback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
    const usb = @ptrCast(*StagingBuffer, @alignCast(@sizeOf(usize), userdata));
    std.debug.assert(usb.slice == null);
    if (status == .success) {
        usb.slice = usb.buffer.getConstMappedRange(f32, 0, 1).?;
    } else {
        std.debug.print("[zgpu] Failed to map buffer (code: {d})\n", .{@enumToInt(status)});
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
    try glfw.init(.{});
    defer glfw.terminate();

    //zgpu.checkSystem(content_dir) catch {
    //    // In case of error zgpu.checkSystem() will print error message.
    //    return;
    //};

    const window = try glfw.Window.create(1280, 960, window_title, null, null, .{
        .client_api = .no_api,
        .cocoa_retina_framebuffer = true,
    });
    defer window.destroy();
    try window.setSizeLimits(.{ .width = 400, .height = 400 }, .{ .width = null, .height = null });

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var demo = try init(allocator, window);
    defer deinit(allocator, &demo);

    zgpu.gui.init(window, demo.gctx.device, content_dir, "Roboto-Medium.ttf", 45.0);
    defer zgpu.gui.deinit();

    while (!window.shouldClose()) {
        try glfw.pollEvents();
        update(&demo);
        draw(&demo);
    }
}
