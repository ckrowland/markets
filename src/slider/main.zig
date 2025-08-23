const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Callbacks = @import("callbacks.zig");
const gui = @import("gui.zig");
const Wgpu = @import("wgpu");
const Consumer = @import("consumer");
const Producer = @import("producer");
const Camera = @import("camera");
const Shapes = @import("shapes");
const emscripten = @import("builtin").target.os.tag == .emscripten;

pub const MAX_NUM_PRODUCERS = 100;
pub const MAX_NUM_CONSUMERS = 10000;
pub const NUM_CONSUMER_SIDES = 40;
pub const PRODUCER_WIDTH = 40;
pub const NUM_STATS = 8;

pub const DemoState = struct {
    gctx: *zgpu.GraphicsContext,
    window: *zglfw.Window,
    allocator: std.mem.Allocator,
    running: bool = false,
    content_scale: f32,
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
            consumers: Wgpu.ObjectBuffer(Consumer),
            consumer_params: zgpu.BufferHandle,
            producers: Wgpu.ObjectBuffer(Producer),
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
    stats: struct {
        num_transactions: std.ArrayList(u32),
        second: f32 = 0,
        num_empty_consumers: std.ArrayList(u32),
        num_total_producer_inventory: std.ArrayList(u32),
        obj_buf: Wgpu.ObjectBuffer(u32),
    },
};

pub const Parameters = struct {
    max_num_stats: u32 = 3,
    num_producers: struct {
        old: u32 = 6,
        new: u32 = 6,
    },
    num_consumers: struct {
        old: u32 = 1000,
        new: u32 = 1000,
    },
    production_rate: u32 = 300,
    demand_rate: i32 = 100,
    max_inventory: u32 = 10000,
    moving_rate: f32 = 5.0,
    consumer_radius: f32 = 6.0,
    num_consumer_sides: u32 = 20,
    aspect: f32,
};

pub fn updateAndRender(demo: *DemoState) !void {
    zglfw.pollEvents();
    zgui.backend.newFrame(
        demo.gctx.swapchain_descriptor.width,
        demo.gctx.swapchain_descriptor.height,
    );
    update(demo);
    draw(demo);
    demo.window.swapBuffers();
}

pub fn setImguiContentScale(scale: f32) void {
    zgui.getStyle().* = zgui.Style.init();
    zgui.getStyle().scaleAllSizes(scale);
}

pub fn getContentScale(window: *zglfw.Window) f32 {
    const content_scale = window.getContentScale();
    if (emscripten) return 1;
    return @max(content_scale[0], content_scale[1]);
}

pub fn deinit(demo: *DemoState) void {
    demo.stats.num_transactions.deinit();
    demo.stats.num_empty_consumers.deinit();
    demo.stats.num_total_producer_inventory.deinit();
    zgui.backend.deinit();
    zgui.plot.deinit();
    zgui.deinit();
    demo.gctx.destroy(demo.allocator);
    demo.window.destroy();
    zglfw.terminate();
}

pub fn init(allocator: std.mem.Allocator) !DemoState {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);

    const window = try zglfw.Window.create(1600, 900, "Simulations", null);
    window.setSizeLimits(400, 400, -1, -1);

    const gctx = try zgpu.GraphicsContext.create(
        allocator,
        .{
            .window = window,
            .fn_getTime = @ptrCast(&zglfw.getTime),
            .fn_getFramebufferSize = @ptrCast(&zglfw.Window.getFramebufferSize),
            .fn_getWin32Window = @ptrCast(&zglfw.getWin32Window),
            .fn_getX11Display = @ptrCast(&zglfw.getX11Display),
            .fn_getX11Window = @ptrCast(&zglfw.getX11Window),
            .fn_getWaylandDisplay = @ptrCast(&zglfw.getWaylandDisplay),
            .fn_getWaylandSurface = @ptrCast(&zglfw.getWaylandWindow),
            .fn_getCocoaWindow = @ptrCast(&zglfw.getCocoaWindow),
        },
        .{},
    );

    zgui.init(allocator);
    zgui.plot.init();
    zgui.io.setIniFilename(null);

    const content_scale = getContentScale(window);
    zgui.getStyle().scaleAllSizes(content_scale);

    _ = zgui.io.addFontFromFile(
        "content/fonts/Roboto-Medium.ttf",
        18 * content_scale,
    );

    zgui.backend.init(
        window,
        gctx.device,
        @intFromEnum(zgpu.GraphicsContext.swapchain_format),
        @intFromEnum(wgpu.TextureFormat.undef),
    );

    const params = Parameters{
        .aspect = Camera.getAspectRatio(gctx),
        .num_producers = .{},
        .num_consumers = .{},
    };

    var consumer_object = Wgpu.createObjectBuffer(
        gctx,
        Consumer,
        MAX_NUM_CONSUMERS,
        0,
    );
    Consumer.generateBulk(
        gctx,
        consumer_object.buf,
        &consumer_object.mapping.num_structs,
        params.aspect,
        params.num_consumers.new,
    );
    const consumer_params_buf = Wgpu.createBuffer(gctx, u32, 3);
    const r = gctx.lookupResource(consumer_params_buf).?;
    gctx.queue.writeBuffer(r, 0, f32, &.{params.moving_rate});
    gctx.queue.writeBuffer(r, @sizeOf(f32), i32, &.{params.demand_rate});

    var producer_object = Wgpu.createObjectBuffer(
        gctx,
        Producer,
        MAX_NUM_PRODUCERS,
        0,
    );
    Producer.generateBulk(
        gctx,
        producer_object.buf,
        &producer_object.mapping.num_structs,
        params.aspect,
        params.num_producers.new,
        params.production_rate,
        params.max_inventory,
    );

    const stats_buf = Wgpu.createObjectBuffer(
        gctx,
        u32,
        NUM_STATS,
        NUM_STATS,
    );
    const resource = gctx.lookupResource(stats_buf.buf).?;
    gctx.queue.writeBuffer(
        resource,
        @sizeOf(u32),
        u32,
        &.{ params.num_consumers.new, params.num_producers.new },
    );

    const compute_bind_group = Wgpu.createComputeBindGroup(gctx, .{
        .consumer = consumer_object.buf,
        .consumer_params = consumer_params_buf,
        .producer = producer_object.buf,
        .stats = stats_buf.buf,
    });
    const depth = Wgpu.createDepthTexture(gctx);

    const bind_group_layout = gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
    });
    const render_bind_group = gctx.createBindGroup(bind_group_layout, &.{
        .{ .binding = 0, .buffer_handle = gctx.uniforms.buffer, .offset = 0, .size = @sizeOf(zm.Mat) },
    });

    return DemoState{
        .gctx = gctx,
        .window = window,
        .content_scale = getContentScale(window),
        .render_pipelines = .{
            .circle = Wgpu.createRenderPipeline(gctx, bind_group_layout, .{
                .vs = @embedFile("shaders/vertex/consumer.wgsl"),
                .inst_type = Consumer,
                .inst_attrs = &.{
                    .{
                        .name = "position",
                        .type = [4]f32,
                    },
                    .{
                        .name = "color",
                        .type = [4]f32,
                    },
                    .{
                        .name = "inventory",
                        .type = u32,
                    },
                },
            }),
            .square = Wgpu.createRenderPipeline(gctx, bind_group_layout, .{
                .vs = @embedFile("shaders/vertex/producer.wgsl"),
                .inst_type = Producer,
                .inst_attrs = &.{
                    .{
                        .name = "home",
                        .type = [4]f32,
                    },
                    .{
                        .name = "color",
                        .type = [4]f32,
                    },
                    .{
                        .name = "inventory",
                        .type = u32,
                    },
                    .{
                        .name = "max_inventory",
                        .type = u32,
                    },
                },
            }),
        },
        .compute_pipelines = .{
            .producer = Wgpu.createComputePipeline(gctx, .{
                .cs = @embedFile("shaders/compute/common.wgsl") ++
                    @embedFile("shaders/compute/producer.wgsl"),
                .entry_point = "main",
            }),
            .consumer = Wgpu.createComputePipeline(gctx, .{
                .cs = @embedFile("shaders/compute/common.wgsl") ++
                    @embedFile("shaders/compute/consumer.wgsl"),
                .entry_point = "main",
            }),
        },
        .bind_groups = .{
            .render = render_bind_group,
            .compute = compute_bind_group,
        },
        .buffers = .{
            .data = .{
                .consumers = consumer_object,
                .consumer_params = consumer_params_buf,
                .producers = producer_object,
            },
            .index = .{
                .circle = Shapes.createCircleIndexBuffer(gctx, NUM_CONSUMER_SIDES),
            },
            .vertex = .{
                .circle = Shapes.createCircleVertexBuffer(
                    gctx,
                    NUM_CONSUMER_SIDES,
                    params.consumer_radius,
                ),
                .square = Shapes.createSquareVertexBuffer(gctx, PRODUCER_WIDTH),
            },
        },
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .params = params,
        .stats = .{
            .num_transactions = std.ArrayList(u32).init(allocator),
            .num_empty_consumers = std.ArrayList(u32).init(allocator),
            .num_total_producer_inventory = std.ArrayList(u32).init(allocator),
            .obj_buf = stats_buf,
        },
    };
}

pub fn draw(demo: *DemoState) void {
    const gctx = demo.gctx;
    const cam_world_to_clip = Camera.getObjectToClipMat(gctx);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        const data = demo.buffers.data;
        const num_consumers = data.consumers.mapping.num_structs;
        const num_producers = data.producers.mapping.num_structs;

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
                pass.dispatchWorkgroups(
                    @divFloor(num_producers, 64) + 1,
                    1,
                    1,
                );
                pass.setPipeline(ccp);
                pass.dispatchWorkgroups(
                    @divFloor(num_consumers, 64) + 1,
                    1,
                    1,
                );
            }
        }

        if (demo.stats.obj_buf.mapping.state == .copy_to_mapped_buffer) {
            pass: {
                const s = gctx.lookupResource(demo.stats.obj_buf.buf) orelse break :pass;
                const s_info = gctx.lookupResourceInfo(demo.stats.obj_buf.buf) orelse break :pass;
                const sm = gctx.lookupResource(demo.stats.obj_buf.mapping.buf) orelse break :pass;
                encoder.copyBufferToBuffer(s, 0, sm, 0, s_info.size);
                demo.stats.obj_buf.mapping.state = .call_map_async;
            }
        }

        if (demo.buffers.data.producers.mapping.state == .copy_to_mapped_buffer) {
            pass: {
                const p = gctx.lookupResource(data.producers.buf) orelse break :pass;
                const p_info = gctx.lookupResourceInfo(data.producers.buf) orelse break :pass;
                const pm = gctx.lookupResource(data.producers.mapping.buf) orelse break :pass;
                encoder.copyBufferToBuffer(p, 0, pm, 0, p_info.size);
                demo.buffers.data.producers.mapping.state = .call_map_async;
            }
        }

        if (demo.buffers.data.consumers.mapping.state == .copy_to_mapped_buffer) {
            pass: {
                const c = gctx.lookupResource(data.consumers.buf) orelse break :pass;
                const c_info = gctx.lookupResourceInfo(data.consumers.buf) orelse break :pass;
                const cm = gctx.lookupResource(data.consumers.mapping.buf) orelse break :pass;
                encoder.copyBufferToBuffer(c, 0, cm, 0, c_info.size);
                demo.buffers.data.consumers.mapping.state = .call_map_async;
            }
        }

        pass: {
            const svb_info = gctx.lookupResourceInfo(demo.buffers.vertex.square) orelse break :pass;
            const pb_info = gctx.lookupResourceInfo(demo.buffers.data.producers.buf) orelse break :pass;
            const cvb_info = gctx.lookupResourceInfo(demo.buffers.vertex.circle) orelse break :pass;
            const cb_info = gctx.lookupResourceInfo(demo.buffers.data.consumers.buf) orelse break :pass;
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
                .stencil_read_only = .true,
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

            const sd = gctx.swapchain_descriptor;
            const width = @as(f32, @floatFromInt(sd.width));
            const xOffset = width / 4;
            const height = @as(f32, @floatFromInt(sd.height));
            const yOffset = height / 4;
            pass.setViewport(xOffset, 0, width - xOffset, height - yOffset, 0, 1);

            var mem = gctx.uniformsAllocate(zm.Mat, 1);
            mem.slice[0] = cam_world_to_clip;
            pass.setBindGroup(0, render_bind_group, &.{mem.offset});

            const num_indices_circle = @as(
                u32,
                @intCast(cib_info.size / @sizeOf(f32)),
            );
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

    if (demo.gctx.present() == .swap_chain_resized) {
        demo.content_scale = getContentScale(demo.window);
        zgui.getStyle().* = zgui.Style.init();
        zgui.getStyle().scaleAllSizes(demo.content_scale);
        updateAspectRatio(demo);
    }
}

pub fn update(demo: *DemoState) void {
    gui.update(demo);
}

pub fn restartSimulation(demo: *DemoState) void {
    if (demo.buffers.data.consumers.mapping.state != .rest or
        demo.buffers.data.producers.mapping.state != .rest or
        demo.stats.obj_buf.mapping.state != .rest)
    {
        while (demo.window.shouldClose() == false) {
            try updateAndRender(demo);
        }
    }

    const encoder = demo.gctx.device.createCommandEncoder(null);
    defer encoder.release();

    Wgpu.clearObjBuffer(encoder, demo.gctx, Consumer, &demo.buffers.data.consumers);
    Wgpu.clearObjBuffer(encoder, demo.gctx, Producer, &demo.buffers.data.producers);
    Wgpu.clearObjBuffer(encoder, demo.gctx, u32, &demo.stats.obj_buf);
    demo.stats.obj_buf.mapping.num_structs = NUM_STATS;

    Consumer.generateBulk(
        demo.gctx,
        demo.buffers.data.consumers.buf,
        &demo.buffers.data.consumers.mapping.num_structs,
        demo.params.aspect,
        demo.params.num_consumers.old,
    );
    Producer.generateBulk(
        demo.gctx,
        demo.buffers.data.producers.buf,
        &demo.buffers.data.producers.mapping.num_structs,
        demo.params.aspect,
        demo.params.num_producers.new,
        demo.params.production_rate,
        demo.params.max_inventory,
    );
    const resource = demo.gctx.lookupResource(demo.stats.obj_buf.buf).?;
    demo.gctx.queue.writeBuffer(
        resource,
        @sizeOf(u32),
        u32,
        &.{ demo.params.num_consumers.new, demo.params.num_producers.new },
    );
    demo.stats.num_transactions.clearAndFree();
    demo.stats.num_empty_consumers.clearAndFree();
    demo.stats.num_total_producer_inventory.clearAndFree();
}

pub fn updateDepthTexture(demo: *DemoState) void {
    // Release old depth texture.
    demo.gctx.releaseResource(demo.depth_texture_view);
    demo.gctx.destroyResource(demo.depth_texture);

    // Create a new depth texture to match the new window size.
    const depth = Wgpu.createDepthTexture(demo.gctx);
    demo.depth_texture = depth.texture;
    demo.depth_texture_view = depth.view;
}

pub fn updateAspectRatio(demo: *DemoState) void {
    updateDepthTexture(demo);
    if (demo.buffers.data.consumers.mapping.state != .rest or
        demo.buffers.data.producers.mapping.state != .rest)
    {
        while (demo.window.shouldClose() == false) {
            try updateAndRender(demo);
        }
    }
    Wgpu.getAllAsync(Consumer, Callbacks.updateConsumerCoords, .{
        .gctx = demo.gctx,
        .obj_buf = &demo.buffers.data.consumers,
    });
    Wgpu.getAllAsync(Producer, Callbacks.updateProducerCoords, .{
        .gctx = demo.gctx,
        .obj_buf = &demo.buffers.data.producers,
    });
    demo.params.aspect = Camera.getAspectRatio(demo.gctx);
}

pub fn main() !void {
    { // Change current working directory to where the executable is located.
        var buffer: [1024]u8 = undefined;
        const path = std.fs.selfExeDirPath(buffer[0..]) catch ".";
        try std.posix.chdir(path);
    }

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var demo = try init(allocator);
    defer deinit(&demo);

    while (demo.window.shouldClose() == false) {
        try updateAndRender(&demo);
    }
}
