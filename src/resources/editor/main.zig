const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const zstbi = @import("zstbi");
const Statistics = @import("../statistics.zig");
const gui = @import("gui.zig");
const Wgpu = @import("../wgpu.zig");
const config = @import("config.zig");
const Consumer = @import("../consumer.zig");
const Producer = @import("../producer.zig");
const ConsumerHover = @import("consumer_hover.zig");
const Image = @import("images.zig");
const Main = @import("../../main.zig");
const Camera = @import("../../camera.zig");
const Square = @import("../../shapes/square.zig");
const Circle = @import("../../shapes/circle.zig");
const Mouse = @import("mouse.zig");
const Hover = @import("hover.zig");
const Popups = @import("popups.zig");
const content_dir = @import("build_options").content_dir;

pub const MAX_NUM_AGENTS = Wgpu.MAX_NUM_STRUCTS;
pub const MAX_NUM_PRODUCERS = 100;
pub const MAX_NUM_CONSUMERS = MAX_NUM_AGENTS;
pub const NUM_CONSUMER_SIDES = 40;
pub const PRODUCER_WIDTH = 40;

const Self = @This();

running: bool = false,
gui: gui.State = .{},
mouse: Mouse.MouseButton = .{},
popups: Popups,
render_pipelines: struct {
    circle: zgpu.RenderPipelineHandle,
    consumer_hover: zgpu.RenderPipelineHandle,
    hover: zgpu.RenderPipelineHandle,
    square: zgpu.RenderPipelineHandle,
},
compute_pipelines: struct {
    consumer: zgpu.ComputePipelineHandle,
    producer: zgpu.ComputePipelineHandle,
},
bind_groups: struct {
    render: zgpu.BindGroupHandle,
    compute: zgpu.BindGroupHandle,
},
buffers: struct {
    data: struct {
        consumer: Wgpu.ObjectBuffer,
        consumer_hover: Wgpu.ObjectBuffer,
        hover: zgpu.BufferHandle,
        producer: Wgpu.ObjectBuffer,
        stats: Wgpu.ObjectBuffer,
    },
    index: struct {
        circle: zgpu.BufferHandle,
    },
    vertex: struct {
        circle: zgpu.BufferHandle,
        hover: zgpu.BufferHandle,
        square: zgpu.BufferHandle,
    },
},
depth_texture: zgpu.TextureHandle,
depth_texture_view: zgpu.TextureViewHandle,
consumer_texture_view: zgpu.TextureViewHandle,
producer_texture_view: zgpu.TextureViewHandle,
params: Parameters,
stats: Statistics,
allocator: std.mem.Allocator,

pub const Parameters = struct {
    max_num_producers: u32 = 100,
    max_num_consumers: u32 = 10000,
    max_num_stats: u32 = 3,
    num_producers: struct {
        old: u32 = 10,
        new: u32 = 10,
    },
    num_consumers: struct {
        old: u32 = 5000,
        new: u32 = 5000,
    },
    production_rate: u32 = 300,
    demand_rate: u32 = 100,
    max_inventory: u32 = 10000,
    moving_rate: f32 = 5.0,
    consumer_radius: f32 = 20.0,
    num_consumer_sides: u32 = 20,
    hover_radius: f32 = 70.0,
    aspect: f32,
};

pub fn init(allocator: std.mem.Allocator, gctx: *zgpu.GraphicsContext) !Self {
    const aspect = Camera.getAspectRatio(gctx);
    const params = Parameters{ .aspect = aspect, .num_producers = .{}, .num_consumers = .{} };

    // Create Buffers
    const hover_buffer = Hover.initBuffer(gctx);
    const consumer_buffer = Wgpu.createBuffer(gctx, Consumer, params.max_num_consumers);
    const consumer_mapped = Wgpu.createMappedBuffer(gctx, Consumer, params.max_num_consumers);
    const consumer_hover = Wgpu.createBuffer(gctx, ConsumerHover, params.max_num_consumers);
    const consumer_h_m = Wgpu.createMappedBuffer(gctx, ConsumerHover, params.max_num_consumers);
    const producer_buffer = Wgpu.createBuffer(gctx, Producer, params.max_num_producers);
    const producer_mapped = Wgpu.createMappedBuffer(gctx, Producer, params.max_num_producers);
    const stats_buffer = Statistics.createBuffer(gctx);
    const stats_mapped = Statistics.createMappedBuffer(gctx);
    const compute_bind_group = Wgpu.createComputeBindGroup(gctx, .{
        .consumer = consumer_buffer,
        .producer = producer_buffer,
        .stats = stats_buffer,
    });
    Statistics.setNumConsumers(gctx, stats_buffer, 0);
    Statistics.setNumProducers(gctx, stats_buffer, 0);

    // Create textures for consumer and producer button images
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    zstbi.init(arena);
    defer zstbi.deinit();
    const consumer_texture_view = try Image.createTextureView(
        gctx,
        content_dir ++ "/pngs/consumerBrush.png",
    );
    const producer_texture_view = try Image.createTextureView(
        gctx,
        content_dir ++ "/pngs/producer.png",
    );
    const depth = Wgpu.createDepthTexture(gctx);

    return Self{
        .render_pipelines = .{
            .circle = Wgpu.createRenderPipeline(gctx, config.cpi),
            .consumer_hover = Wgpu.createRenderPipeline(gctx, config.chpi),
            .hover = Wgpu.createRenderPipeline(gctx, config.hpi),
            .square = Wgpu.createRenderPipeline(gctx, config.ppi),
        },
        .compute_pipelines = .{
            .producer = Wgpu.createComputePipeline(gctx, config.pcpi),
            .consumer = Wgpu.createComputePipeline(gctx, config.ccpi),
        },
        .bind_groups = .{
            .render = Wgpu.createUniformBindGroup(gctx),
            .compute = compute_bind_group,
        },
        .buffers = .{
            .data = .{
                .consumer = .{
                    .data = consumer_buffer,
                    .mapped = consumer_mapped,
                },
                .consumer_hover = .{
                    .data = consumer_hover,
                    .mapped = consumer_h_m,
                },
                .hover = hover_buffer,
                .producer = .{
                    .data = producer_buffer,
                    .mapped = producer_mapped,
                },
                .stats = .{
                    .data = stats_buffer,
                    .mapped = stats_mapped,
                },
            },
            .index = .{
                .circle = Circle.createIndexBuffer(gctx, 40),
            },
            .vertex = .{
                .circle = Circle.createVertexBuffer(gctx, 40, params.consumer_radius),
                .hover = Circle.createVertexBuffer(gctx, 40, params.hover_radius),
                .square = Square.createVertexBuffer(gctx, 40),
            },
        },
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .consumer_texture_view = consumer_texture_view,
        .producer_texture_view = producer_texture_view,
        .popups = Popups.init(allocator),
        .allocator = allocator,
        .params = params,
        .stats = Statistics.init(allocator),
    };
}

pub fn deinit(demo: *Self) void {
    demo.popups.deinit();
    demo.stats.deinit();
    demo.* = undefined;
}

pub fn update(demo: *Self, gctx: *zgpu.GraphicsContext) !void {
    demo.mouse.update(gctx);
    try gui.update(demo, gctx);
}

pub fn draw(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    const cam_world_to_clip = Camera.getObjectToClipMat(gctx);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    
    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        const data = demo.buffers.data;
        const num_consumers = Wgpu.getNumStructs(gctx, Consumer, demo.buffers.data.stats);
        const num_producers = Wgpu.getNumStructs(gctx, Producer, demo.buffers.data.stats);

        // Compute shaders
        if (demo.running) {
            pass: {
                const pcp = gctx.lookupResource(demo.compute_pipelines.producer) orelse break :pass;
                const ccp = gctx.lookupResource(demo.compute_pipelines.consumer) orelse break :pass;
                const bg = gctx.lookupResource(demo.bind_groups.compute) orelse break :pass;

                const pass = encoder.beginComputePass(null);
                defer {
                    pass.end();
                    pass.release();
                }
                pass.setBindGroup(0, bg, &.{});

                pass.setPipeline(pcp);
                pass.dispatchWorkgroups(@divFloor(num_producers, 64) + 1, 1, 1);

                pass.setPipeline(ccp);
                pass.dispatchWorkgroups(@divFloor(num_consumers, 64) + 1, 1, 1);
            }
        }

        // Copy data to mapped buffers so we can retrieve it on demand
        pass: {
            const s = gctx.lookupResource(data.stats.data) orelse break :pass;
            const s_info = gctx.lookupResourceInfo(data.stats.data) orelse break :pass;
            const sm = gctx.lookupResource(data.stats.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(s, 0, sm, 0, s_info.size);

            const p = gctx.lookupResource(data.producer.data) orelse break :pass;
            const p_info = gctx.lookupResourceInfo(data.producer.data) orelse break :pass;
            const pm = gctx.lookupResource(data.producer.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(p, 0, pm, 0, p_info.size);

            const c = gctx.lookupResource(data.consumer.data) orelse break :pass;
            const c_info = gctx.lookupResourceInfo(data.consumer.data) orelse break :pass;
            const cm = gctx.lookupResource(data.consumer.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(c, 0, cm, 0, c_info.size);
            
            const ch = gctx.lookupResource(data.consumer_hover.data) orelse break :pass;
            const ch_info = gctx.lookupResourceInfo(data.consumer_hover.data) orelse break :pass;
            const chm = gctx.lookupResource(data.consumer_hover.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(ch, 0, chm, 0, ch_info.size);
        }

        // Draw the circles and squares in our defined viewport
        pass: {
            const hoverRP = gctx.lookupResource(demo.render_pipelines.hover) orelse break :pass;
            const hoverVB = gctx.lookupResourceInfo(demo.buffers.vertex.hover) orelse break :pass;
            const hoverB = gctx.lookupResourceInfo(data.hover) orelse break :pass;
            const svb_info = gctx.lookupResourceInfo(demo.buffers.vertex.square) orelse break :pass;
            const pb_info = gctx.lookupResourceInfo(data.producer.data) orelse break :pass;
            const cvb_info = gctx.lookupResourceInfo(demo.buffers.vertex.circle) orelse break :pass;
            const cb_info = gctx.lookupResourceInfo(data.consumer.data) orelse break :pass;
            const cib_info = gctx.lookupResourceInfo(demo.buffers.index.circle) orelse break :pass;
            const square_rp = gctx.lookupResource(demo.render_pipelines.square) orelse break :pass;
            const circle_rp = gctx.lookupResource(demo.render_pipelines.circle) orelse break :pass;
            const chrp = gctx.lookupResource(demo.render_pipelines.consumer_hover) orelse break :pass;
            const ch_info = gctx.lookupResourceInfo(data.consumer_hover.data) orelse break :pass;
            const render_bind_group = gctx.lookupResource(demo.bind_groups.render) orelse break :pass;
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

            const width = @floatFromInt(f32, gctx.swapchain_descriptor.width);
            const xOffset = width / 4;
            const height = @floatFromInt(f32, gctx.swapchain_descriptor.height);
            const yOffset = height / 4;
            pass.setViewport(xOffset, 0, width - xOffset, height - yOffset, 0, 1);

            var mem = gctx.uniformsAllocate(zm.Mat, 1);
            mem.slice[0] = cam_world_to_clip;
            pass.setBindGroup(0, render_bind_group, &.{mem.offset});

            const num_indices_circle = @intCast(u32, cib_info.size / @sizeOf(f32));
            pass.setPipeline(hoverRP);
            pass.setVertexBuffer(0, hoverVB.gpuobj.?, 0, hoverVB.size);
            pass.setVertexBuffer(1, hoverB.gpuobj.?, 0, hoverB.size);
            pass.setIndexBuffer(cib_info.gpuobj.?, .uint32, 0, cib_info.size);
            pass.drawIndexed(num_indices_circle, 1, 0, 0, 0);

            pass.setPipeline(chrp);
            pass.setVertexBuffer(0, hoverVB.gpuobj.?, 0, hoverVB.size);
            pass.setVertexBuffer(1, ch_info.gpuobj.?, 0, ch_info.size);
            pass.setIndexBuffer(cib_info.gpuobj.?, .uint32, 0, cib_info.size);
            pass.drawIndexed(num_indices_circle, num_consumers, 0, 0, 0);

            pass.setPipeline(circle_rp);
            pass.setVertexBuffer(0, cvb_info.gpuobj.?, 0, cvb_info.size);
            pass.setVertexBuffer(1, cb_info.gpuobj.?, 0, cb_info.size);
            pass.setIndexBuffer(cib_info.gpuobj.?, .uint32, 0, cib_info.size);
            pass.drawIndexed(num_indices_circle, num_consumers, 0, 0, 0);

            pass.setPipeline(square_rp);
            pass.setVertexBuffer(0, svb_info.gpuobj.?, 0, svb_info.size);
            pass.setVertexBuffer(1, pb_info.gpuobj.?, 0, pb_info.size);
            pass.draw(6, num_producers, 0, 0);
            
        }

        // Draw ImGui
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
        demo.updateDepthTexture(gctx);

        // Update grid positions to new aspect ratio
        const aspect = Camera.getAspectRatio(gctx);
        demo.params.aspect = aspect;
        Wgpu.updateCoords(gctx, Consumer, .{
            .structs = demo.buffers.data.consumer,
            .stats = demo.buffers.data.stats,
        });
        Wgpu.updateCoords(gctx, Producer, .{
            .structs = demo.buffers.data.producer,
            .stats = demo.buffers.data.stats,
        });
        Wgpu.updateCoords(gctx, ConsumerHover, .{
            .structs = demo.buffers.data.consumer_hover,
            .stats = demo.buffers.data.stats,
        });
    }
}

pub fn restartSimulation(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    Wgpu.clearBuffer(gctx, demo.buffers.data.consumer.data);
    Wgpu.clearBuffer(gctx, demo.buffers.data.producer.data);
    Wgpu.clearBuffer(gctx, demo.buffers.data.consumer_hover.data);
    Wgpu.clearBuffer(gctx, demo.buffers.data.stats.data);
    Statistics.setNumConsumers(gctx, demo.buffers.data.stats.data, 0);
    Statistics.setNumProducers(gctx, demo.buffers.data.stats.data, 0);
    demo.stats.clear();
    demo.popups.clear();
    demo.running = false;
}

pub fn updateDepthTexture(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    // Release old depth texture.
    gctx.releaseResource(demo.depth_texture_view);
    gctx.destroyResource(demo.depth_texture);

    // Create a new depth texture to match the new window size.
    const depth = Wgpu.createDepthTexture(gctx);
    demo.depth_texture = depth.texture;
    demo.depth_texture_view = depth.view;
}

