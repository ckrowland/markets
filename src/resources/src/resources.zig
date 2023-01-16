const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Statistics = @import("statistics.zig");
const gui = @import("gui.zig");
const Wgpu = @import("wgpu.zig");
const config = @import("config.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Circle = @import("circle.zig");
const Square = @import("square.zig");

const content_dir = @import("build_options").content_dir;
const window_title = "Resource Simulation";

pub const Parameters = struct {
    num_producers: u32 = 10,
    production_rate: u32 = 300,
    demand_rate: u32 = 100,
    max_inventory: u32 = 10000,
    num_consumers: u32 = 5000,
    moving_rate: f32 = 5.0,
    producer_width: f32 = 40.0,
    consumer_radius: f32 = 20.0,
    num_consumer_sides: u32 = 20,
};

pub const CoordinateSize = struct {
    min_x: i32 = -1000,
    min_y: i32 = -500,
    max_x: i32 = 1800,
    max_y: i32 = 1200,
};

pub const DemoState = struct {
    gctx: *zgpu.GraphicsContext,
    running: bool,
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
            consumer: zgpu.BufferHandle,
            consumer_mapped: zgpu.BufferHandle,
            producer: zgpu.BufferHandle,
            producer_mapped: zgpu.BufferHandle,
            stats: zgpu.BufferHandle,
            stats_mapped: zgpu.BufferHandle,
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
    coordinate_size: CoordinateSize,
    stats: Statistics,
    allocator: std.mem.Allocator,
};

fn init(allocator: std.mem.Allocator, window: *zglfw.Window) !DemoState {
    const gctx = try zgpu.GraphicsContext.create(allocator, window);

    const params = Parameters{};
    const coordinate_size = CoordinateSize{};

    // Create Buffers
    const consumer_buffer = Consumer.generateBuffer(gctx, params, coordinate_size);
    const producer_buffer = Producer.generateBuffer(gctx, params, coordinate_size);
    const stats_buffer = Statistics.createBuffer(gctx);

    const compute_bind_group = Wgpu.createComputeBindGroup(gctx, consumer_buffer, producer_buffer, stats_buffer);

    // Create a depth texture and its 'view'.
    const depth = createDepthTexture(gctx);

    return DemoState{
        .gctx = gctx,
        .running = false,
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
                .consumer = consumer_buffer,
                .consumer_mapped = Wgpu.createMappedBuffer(gctx, Consumer, params.num_consumers),
                .producer = producer_buffer,
                .producer_mapped = Wgpu.createMappedBuffer(gctx, Producer, params.num_consumers),
                .stats = stats_buffer,
                .stats_mapped = Statistics.createMappedBuffer(gctx),
            },
            .index = .{
                .circle = Circle.createIndexBuffer(gctx),
            },
            .vertex = .{
                .circle = Circle.createVertexBuffer(gctx, params.consumer_radius),
                .square = Square.createVertexBuffer(gctx, params.producer_width),
            },
        },
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .coordinate_size = coordinate_size,
        .params = params,
        .stats = Statistics.init(allocator),
    };
}

fn deinit(allocator: std.mem.Allocator, demo: *DemoState) void {
    demo.gctx.destroy(allocator);
    demo.stats.deinit();
    demo.* = undefined;
}

fn update(demo: *DemoState) void {
    gui.update(demo);
}

fn draw(demo: *DemoState) void {
    const gctx = demo.gctx;

    const cam_world_to_view = zm.lookAtLh(
        //eye position
        zm.f32x4(0.0, 0.0, -3000.0, 0.0),

        //focus position
        zm.f32x4(0.0, 0.0, 0.0, 0.0),

        //up direction
        zm.f32x4(0.0, 1.0, 0.0, 0.0),
    );

    const cam_view_to_clip = zm.perspectiveFovLh(
        //fovy
        0.25 * math.pi,

        //aspect
        1.8,

        //near
        0.01,

        //far
        3001.0,
    );
    const cam_world_to_clip = zm.mul(cam_world_to_view, cam_view_to_clip);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        if (demo.running) {
            pass: {
                const pb_info = gctx.lookupResourceInfo(demo.buffers.data.producer) orelse break :pass;
                const cb_info = gctx.lookupResourceInfo(demo.buffers.data.consumer) orelse break :pass;
                const pcp = gctx.lookupResource(demo.compute_pipelines.producer) orelse break :pass;
                const ccp = gctx.lookupResource(demo.compute_pipelines.consumer) orelse break :pass;
                const bg = gctx.lookupResource(demo.bind_groups.compute) orelse break :pass;

                const num_consumers = @intCast(u32, cb_info.size / @sizeOf(Consumer));
                const num_producers = @intCast(u32, pb_info.size / @sizeOf(Producer));

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

        // Copy data to mapped buffers so we can retrieve it
        pass: {
            const s = gctx.lookupResource(demo.buffers.data.stats) orelse break :pass;
            const s_info = gctx.lookupResourceInfo(demo.buffers.data.stats) orelse break :pass;
            const sm = gctx.lookupResource(demo.buffers.data.stats_mapped) orelse break :pass;
            encoder.copyBufferToBuffer(s, 0, sm, 0, s_info.size);

            const p = gctx.lookupResource(demo.buffers.data.producer) orelse break :pass;
            const p_info = gctx.lookupResourceInfo(demo.buffers.data.producer) orelse break :pass;
            const pm = gctx.lookupResource(demo.buffers.data.producer_mapped) orelse break :pass;
            encoder.copyBufferToBuffer(p, 0, pm, 0, p_info.size);

            const c = gctx.lookupResource(demo.buffers.data.consumer) orelse break :pass;
            const c_info = gctx.lookupResourceInfo(demo.buffers.data.consumer) orelse break :pass;
            const cm = gctx.lookupResource(demo.buffers.data.consumer_mapped) orelse break :pass;
            encoder.copyBufferToBuffer(c, 0, cm, 0, c_info.size);
        }

        pass: {
            const svb_info = gctx.lookupResourceInfo(demo.buffers.vertex.square) orelse break :pass;
            const pb_info = gctx.lookupResourceInfo(demo.buffers.data.producer) orelse break :pass;
            const cvb_info = gctx.lookupResourceInfo(demo.buffers.vertex.circle) orelse break :pass;
            const cb_info = gctx.lookupResourceInfo(demo.buffers.data.consumer) orelse break :pass;
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
            const num_consumers = @intCast(u32, cb_info.size / @sizeOf(Consumer));
            const num_producers = @intCast(u32, pb_info.size / @sizeOf(Producer));

            var mem = gctx.uniformsAllocate(zm.Mat, 1);
            mem.slice[0] = zm.transpose(cam_world_to_clip);
            pass.setBindGroup(0, render_bind_group, &.{mem.offset});

            pass.setPipeline(circle_rp);
            pass.setVertexBuffer(0, cvb_info.gpuobj.?, 0, cvb_info.size);
            pass.setVertexBuffer(1, cb_info.gpuobj.?, 0, cb_info.size);
            pass.setIndexBuffer(cib_info.gpuobj.?, .uint32, 0, cib_info.size);
            pass.drawIndexed(57, num_consumers, 0, 0, 0);

            pass.setPipeline(square_rp);
            pass.setVertexBuffer(0, svb_info.gpuobj.?, 0, svb_info.size);
            pass.setVertexBuffer(1, pb_info.gpuobj.?, 0, pb_info.size);
            pass.draw(6, num_producers, 0, 0);
        }

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

pub fn restartSimulation(demo: *DemoState) void {
    demo.buffers.data.consumer = Consumer.generateBuffer(demo.gctx, demo.params, demo.coordinate_size);
    demo.buffers.data.producer = Producer.generateBuffer(demo.gctx, demo.params, demo.coordinate_size);

    demo.stats.clear();
    Statistics.clearStatsBuffer(demo.gctx, demo.buffers.data.stats);

    demo.bind_groups.compute = Wgpu.createComputeBindGroup(demo.gctx, demo.buffers.data.consumer, demo.buffers.data.producer, demo.buffers.data.stats);
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

    //zglfw.Hint.reset();
    //zglfw.Hint.set(.cocoa_retina_framebuffer, 1);
    //zglfw.Hint.set(.client_api, 0);
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
        if (!window.getAttribute(.focused)) {
            continue;
        }
        update(&demo);
        draw(&demo);
    }
}
