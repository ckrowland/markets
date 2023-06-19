const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Statistics = @import("statistics.zig");
const gui = @import("gui.zig");
const Wgpu = @import("../wgpu.zig");
const config = @import("config.zig");
const Consumer = @import("../consumer.zig");
const Producer = @import("../producer.zig");
const Main = @import("../../main.zig");
const Camera = @import("../../camera.zig");
const Square = @import("../../shapes/square.zig");
const Circle = @import("../../shapes/circle.zig");

const content_dir = @import("build_options").content_dir;

pub const MAX_NUM_PRODUCERS = 100;
pub const MAX_NUM_CONSUMERS = 10000;
pub const NUM_CONSUMER_SIDES = 40;
pub const PRODUCER_WIDTH = 40;

pub const Parameters = struct {
    max_num_stats: u32 = 3,
    num_producers: struct {
        old: u32 = 6,
        new: u32 = 6,
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
    aspect: f32,
};


const Self = @This();

running: bool = false,
render_pipelines: struct {
    circle: zgpu.RenderPipelineHandle,
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
        producer: Wgpu.ObjectBuffer,
        stats: Wgpu.ObjectBuffer,
    },
    index: struct {
        circle: zgpu.BufferHandle,
    },
    vertex: struct {
        circle: zgpu.BufferHandle,
        square: zgpu.BufferHandle,
    },
},
depth_texture: zgpu.TextureHandle,
depth_texture_view: zgpu.TextureViewHandle,
params: Parameters,
stats: Statistics,
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, gctx: *zgpu.GraphicsContext) !Self {
    const aspect = Camera.getAspectRatio(gctx);
    const params = Parameters{ .aspect = aspect, .num_producers = .{}, .num_consumers = .{}};

    const consumer_buffer = Wgpu.createBuffer(gctx, Consumer, MAX_NUM_CONSUMERS);
    const consumer_mapped = Wgpu.createMappedBuffer(gctx, Consumer, MAX_NUM_CONSUMERS);
    Consumer.generateBulk(gctx, consumer_buffer, params);
    
    const producer_buffer = Wgpu.createBuffer(gctx, Producer, MAX_NUM_PRODUCERS);
    const producer_mapped = Wgpu.createMappedBuffer(gctx, Producer, MAX_NUM_PRODUCERS);
    Producer.generateBulk(gctx, producer_buffer, params);
    
    const stats_buffer = Statistics.createBuffer(gctx);
    const stats_mapped = Statistics.createMappedBuffer(gctx);
    Statistics.setNumConsumers(gctx, stats_buffer, params.num_consumers.new);
    Statistics.setNumProducers(gctx, stats_buffer, params.num_producers.new);
    
    const compute_bind_group = Wgpu.createComputeBindGroup(gctx, .{
        .consumer = consumer_buffer,
        .producer = producer_buffer,
        .stats = stats_buffer,
    });
    const depth = Main.createDepthTexture(gctx);
    
    return Self{
        .render_pipelines = .{
            .circle = Wgpu.createRenderPipeline(gctx, config.cpi),
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
                .circle = Circle.createIndexBuffer(gctx, NUM_CONSUMER_SIDES),
            },
            .vertex = .{
                .circle = Circle.createVertexBuffer(
                    gctx,
                    NUM_CONSUMER_SIDES,
                    params.consumer_radius,
                ),
                .square = Square.createVertexBuffer(gctx, PRODUCER_WIDTH),
            },
        },
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .params = params,
        .stats = Statistics.init(allocator),
    };
}

pub fn deinit(demo: *Self) void {
    demo.stats.deinit();
    demo.* = undefined;
}

pub fn update(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    gui.update(demo, gctx);
}

pub fn draw(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    const cam_world_to_clip = Camera.getObjectToClipMat(gctx);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();
        
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
                pass.dispatchWorkgroups(@divFloor(num_producers, 64) + 1, 1, 1,);

                pass.setPipeline(ccp);
                pass.dispatchWorkgroups(@divFloor(num_consumers, 64) + 1, 1, 1,);
            }
        }

        // Copy data to mapped buffers so we can retrieve it on demand
        pass: {
            const s = gctx.lookupResource(demo.buffers.data.stats.data) orelse break :pass;
            const s_info = gctx.lookupResourceInfo(demo.buffers.data.stats.data) orelse break :pass;
            const sm = gctx.lookupResource(demo.buffers.data.stats.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(s, 0, sm, 0, s_info.size);

            const p = gctx.lookupResource(demo.buffers.data.producer.data) orelse break :pass;
            const p_info = gctx.lookupResourceInfo(demo.buffers.data.producer.data) orelse break :pass;
            const pm = gctx.lookupResource(demo.buffers.data.producer.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(p, 0, pm, 0, p_info.size);

            const c = gctx.lookupResource(demo.buffers.data.consumer.data) orelse break :pass;
            const c_info = gctx.lookupResourceInfo(demo.buffers.data.consumer.data) orelse break :pass;
            const cm = gctx.lookupResource(demo.buffers.data.consumer.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(c, 0, cm, 0, c_info.size);
        }

        // Draw the circles and squares in our defined viewport
        pass: {
            const svb_info = gctx.lookupResourceInfo(demo.buffers.vertex.square) orelse break :pass;
            const pb_info = gctx.lookupResourceInfo(demo.buffers.data.producer.data) orelse break :pass;
            const cvb_info = gctx.lookupResourceInfo(demo.buffers.vertex.circle) orelse break :pass;
            const cb_info = gctx.lookupResourceInfo(demo.buffers.data.consumer.data) orelse break :pass;
            const cib_info = gctx.lookupResourceInfo(demo.buffers.index.circle) orelse break :pass;
            const square_rp = gctx.lookupResource(demo.render_pipelines.square) orelse break :pass;
            const circle_rp = gctx.lookupResource(demo.render_pipelines.circle) orelse break :pass;
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

            const width = @intToFloat(f32, gctx.swapchain_descriptor.width);
            const xOffset = width / 4;
            const height = @intToFloat(f32, gctx.swapchain_descriptor.height);
            const yOffset = height / 4;
            pass.setViewport(xOffset, 0, width - xOffset, height - yOffset, 0, 1);

            var mem = gctx.uniformsAllocate(zm.Mat, 1);
            mem.slice[0] = zm.transpose(cam_world_to_clip);
            pass.setBindGroup(0, render_bind_group, &.{mem.offset});
            
            const num_indices_circle = @intCast(u32, cib_info.size / @sizeOf(f32));
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
        Consumer.updateCoords(gctx, .{
            .consumers = demo.buffers.data.consumer,
            .stats = demo.buffers.data.stats,
        });
        Producer.updateCoords(gctx, .{
            .producers = demo.buffers.data.producer,
            .stats = demo.buffers.data.stats,
        });
    }
}

pub fn restartSimulation(demo: *Self, gctx: *zgpu.GraphicsContext) void {
    Consumer.generateBulk(gctx, demo.buffers.data.consumer.data, demo.params);
    Producer.generateBulk(gctx, demo.buffers.data.producer.data, demo.params);
    demo.stats.clear();
    Statistics.clearNumTransactions(gctx, demo.buffers.data.stats.data);
}

pub fn updateDepthTexture(state: *Self, gctx: *zgpu.GraphicsContext) void {
    // Release old depth texture.
    gctx.releaseResource(state.depth_texture_view);
    gctx.destroyResource(state.depth_texture);

    // Create a new depth texture to match the new window size.
    const depth = Main.createDepthTexture(gctx);
    state.depth_texture = depth.texture;
    state.depth_texture_view = depth.view;
}
