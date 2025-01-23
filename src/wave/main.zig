const std = @import("std");
const math = std.math;
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Statistics = @import("statistics");
const Gui = @import("gui.zig");
const Wgpu = @import("wgpu");
const Consumer = @import("consumer");
const Producer = @import("producer");
const Camera = @import("camera");
const Shapes = @import("shapes");
const Callbacks = @import("callbacks.zig");
const zemscripten = @import("zemscripten");
const emscripten = @import("builtin").target.os.tag == .emscripten;

pub const MAX_NUM_PRODUCERS = 100;
pub const MAX_NUM_CONSUMERS = 40000;
pub const NUM_CONSUMER_SIDES = 40;
pub const PRODUCER_WIDTH = 40;
pub const CONSUMER_RADIUS: f32 = 1.0;

pub const DemoState = struct {
    gctx: *zgpu.GraphicsContext,
    window: *zglfw.Window,
    aspect: f32,
    allocator: std.mem.Allocator,
    running: bool = false,
    push_coord_update: bool = false,
    push_restart: bool = false,
    content_scale: f32,
    timeline_visible: bool,
    params: struct {
        sample_idx: usize,
        radian: f64,
        max_num_stats: u32,
        num_consumer_sides: u32,
        plot_hovered: bool,
    },
    sliders: std.StringArrayHashMap(Gui.Variable(u32)),
    f_sliders: std.StringArrayHashMap(Gui.Variable(f32)),
    stats: Statistics,
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
};

//pub const Sliders = struct {
//    production_rate: Gui.Variable(u32),
//    max_inventory: Gui.Variable(u32),
//    consumer_size: Gui.Variable(f32),
//    num_consumers: Gui.Variable(u32),
//    num_producers: Gui.Variable(u32),
//    demand_rate: Gui.Variable(u32),
//    moving_rate: Gui.Variable(f32),
//    income: Gui.Variable(u32),
//};

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

pub fn deinit(demo: *DemoState) void {
    const allocator = demo.allocator;
    demo.sliders.getPtr("num_consumers").?.deinit(allocator);
    demo.sliders.getPtr("num_producers").?.deinit(allocator);
    demo.sliders.getPtr("demand_rate").?.deinit(allocator);
    demo.sliders.getPtr("production_rate").?.deinit(allocator);
    demo.sliders.getPtr("max_inventory").?.deinit(allocator);
    demo.f_sliders.getPtr("moving_rate").?.deinit(allocator);
    demo.f_sliders.getPtr("consumer_size").?.deinit(allocator);
    demo.sliders.deinit();
    demo.f_sliders.deinit();
    demo.stats.deinit();

    zgui.backend.deinit();
    zgui.plot.deinit();
    zgui.deinit();
    demo.gctx.destroy(allocator);
    demo.window.destroy();
    zglfw.terminate();
}

pub fn init(allocator: std.mem.Allocator) !DemoState {
    try zglfw.init();
    zglfw.windowHintTyped(.client_api, .no_api);

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

    const consumer_object = Wgpu.createObjectBuffer(
        gctx,
        Consumer,
        MAX_NUM_CONSUMERS,
        0,
    );
    const consumer_params_buf = Wgpu.createBuffer(gctx, u32, 3);
    const producer_object = Wgpu.createObjectBuffer(
        gctx,
        Producer,
        MAX_NUM_PRODUCERS,
        0,
    );

    var stats = Statistics.init(gctx, allocator);
    stats.setNum(gctx, 0, .consumers);
    stats.setNum(gctx, 0, .producers);
    const compute_bind_group = Wgpu.createComputeBindGroup(gctx, .{
        .consumer = consumer_object.buf,
        .consumer_params = consumer_params_buf,
        .producer = producer_object.buf,
        .stats = stats.obj_buf.buf,
    });

    const bind_group_layout = gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
    });
    const render_bind_group = gctx.createBindGroup(bind_group_layout, &.{
        .{ .binding = 0, .buffer_handle = gctx.uniforms.buffer, .offset = 0, .size = @sizeOf(zm.Mat) },
    });
    const depth = Wgpu.createDepthTexture(gctx);
    var slider_map = std.StringArrayHashMap(Gui.Variable(u32)).init(allocator);
    try slider_map.put(
        "num_consumers",
        Gui.Variable(u32).init(allocator, .{
            .min = 0,
            .mid = 5000,
            .max = 10000,
            .margin = 100,
            .ratio = 1,
            .variable = true,
            .extra = false,
            .title = "Number of Consumers",
            .agent = .{ .custom = Gui.numConsumerUpdate },
        }),
    );
    try slider_map.put(
        "num_producers",
        Gui.Variable(u32).init(allocator, .{
            .min = 1,
            .mid = 10,
            .max = 100,
            .margin = 10,
            .ratio = 0.7,
            .variable = false,
            .extra = false,
            .title = "Number of Producers",
            .agent = .{ .custom = Gui.numProducersUpdate },
        }),
    );
    try slider_map.put(
        "demand_rate",
        Gui.Variable(u32).init(allocator, .{
            .min = 1,
            .mid = 100,
            .max = 300,
            .margin = 100,
            .ratio = 0.9,
            .variable = false,
            .extra = true,
            .title = "Max Demand Rate",
            .help = "The maximum amount a consumer will buy given they have enough money.",
            .agent = .{ .consumer = @offsetOf(Consumer.Params, "demand_rate") },
        }),
    );
    try slider_map.put(
        "max_inventory",
        Gui.Variable(u32).init(allocator, .{
            .min = 100,
            .mid = 50000,
            .max = 100000,
            .margin = 100,
            .ratio = 1,
            .variable = false,
            .extra = true,
            .title = "Max Producer Inventory",
            .help = "The maximum number of resources a producer can hold at one time.",
            .agent = .{ .producer = @offsetOf(Producer, "max_inventory") },
        }),
    );
    try slider_map.put(
        "production_rate",
        Gui.Variable(u32).init(allocator, .{
            .min = 1,
            .mid = 300,
            .max = 1000,
            .margin = 100,
            .ratio = 1.4,
            .variable = false,
            .extra = false,
            .title = "Production Rate",
            .help = "How many resources 1 producer creates each frame.",
            .agent = .{ .producer = @offsetOf(Producer, "production_rate") },
        }),
    );

    var f_slider_map = std.StringArrayHashMap(Gui.Variable(f32)).init(allocator);
    try f_slider_map.put(
        "moving_rate",
        Gui.Variable(f32).init(allocator, .{
            .min = 1,
            .mid = 5,
            .max = 20,
            .margin = 100,
            .ratio = 1,
            .extra = true,
            .variable = false,
            .title = "Moving Rate",
            .agent = .{ .consumer = @offsetOf(Consumer.Params, "moving_rate") },
        }),
    );
    try f_slider_map.put(
        "consumer_size",
        Gui.Variable(f32).init(allocator, .{
            .min = 0,
            .mid = 1,
            .max = 3,
            .margin = 100,
            .ratio = 0.5,
            .extra = true,
            .variable = false,
            .title = "Consumer Size",
            .agent = .{ .custom = Gui.consumerSize },
        }),
    );
    var demo = DemoState{
        .gctx = gctx,
        .window = window,
        .timeline_visible = true,
        .aspect = Camera.getAspectRatio(gctx),
        .content_scale = getContentScale(window),
        .sliders = slider_map,
        .f_sliders = f_slider_map,
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
                    CONSUMER_RADIUS,
                ),
                .square = Shapes.createSquareVertexBuffer(gctx, PRODUCER_WIDTH),
            },
        },
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .params = .{
            .sample_idx = 0,
            .radian = 0,
            .max_num_stats = 3,
            .plot_hovered = false,
            .num_consumer_sides = 20,
        },
        .stats = stats,
    };

    const num_consumers = slider_map.get("num_consumers").?.slider.val;
    const num_producers = slider_map.get("num_producers").?.slider.val;
    demo.stats.setNum(gctx, num_consumers, .consumers);
    demo.stats.setNum(gctx, num_producers, .producers);
    Consumer.generateBulk(
        demo.gctx,
        demo.buffers.data.consumers.buf,
        &demo.buffers.data.consumers.mapping.num_structs,
        demo.aspect,
        num_consumers,
    );
    Consumer.setParamsBuf(
        demo.gctx,
        demo.buffers.data.consumer_params,
        f_slider_map.get("moving_rate").?.slider.val,
        slider_map.get("demand_rate").?.slider.val,
    );

    Producer.generateBulk(
        demo.gctx,
        demo.buffers.data.producers.buf,
        &demo.buffers.data.producers.mapping.num_structs,
        demo.aspect,
        num_producers,
        slider_map.get("production_rate").?.slider.val,
        slider_map.get("max_inventory").?.slider.val,
    );
    setImguiContentScale(demo.content_scale);

    return demo;
}

pub fn update(demo: *DemoState) void {
    if (demo.push_restart) restartSimulation(demo);
    if (demo.push_coord_update) updateAspectRatio(demo);

    Wgpu.checkObjBufState(u32, &demo.stats.obj_buf.mapping);
    Wgpu.checkObjBufState(Producer, &demo.buffers.data.producers.mapping);
    Wgpu.checkObjBufState(Consumer, &demo.buffers.data.consumers.mapping);

    //zgui.showDemoWindow(null);
    Gui.update(demo);

    if (demo.running) {
        Gui.updateWaves(demo);

        //Statistics.generateAndFillRandomColor(
        //    demo.gctx,
        //    demo.buffers.data.stats.buf,
        //);
        const current_time = @as(f32, @floatCast(demo.gctx.stats.time));
        const seconds_passed = current_time - demo.stats.second;
        if (seconds_passed >= 1) {
            demo.stats.second = current_time;
            Wgpu.getAllAsync(u32, Callbacks.numTransactions, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.stats.obj_buf,
                .stat_array = &demo.stats.num_transactions,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.consumerStats, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.consumers,
                .stat_array = &demo.stats.num_empty_consumers,
            });
            Wgpu.getAllAsync(Producer, Callbacks.producerStats, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.producers,
                .stat_array = &demo.stats.num_total_producer_inventory,
            });
        }
    }
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
        setImguiContentScale(demo.content_scale);
        updateAspectRatio(demo);
    }
}

pub fn restartSimulation(demo: *DemoState) void {
    const consumer_waiting = demo.buffers.data.consumers.mapping.state != .rest;
    const producer_waiting = demo.buffers.data.producers.mapping.state != .rest;
    const stats_waiting = demo.stats.obj_buf.mapping.state != .rest;
    if (consumer_waiting or producer_waiting or stats_waiting) {
        demo.push_restart = true;
        return;
    }

    const encoder = demo.gctx.device.createCommandEncoder(null);
    defer encoder.release();

    Wgpu.clearObjBuffer(encoder, demo.gctx, Consumer, &demo.buffers.data.consumers);
    Wgpu.clearObjBuffer(encoder, demo.gctx, Producer, &demo.buffers.data.producers);
    Wgpu.clearObjBuffer(encoder, demo.gctx, u32, &demo.stats.obj_buf);
    demo.stats.obj_buf.mapping.num_structs = Statistics.NUM_STATS;

    const num_consumers = demo.sliders.get("num_consumers").?.slider.val;
    const num_producers = demo.sliders.get("num_producers").?.slider.val;

    demo.stats.setNum(demo.gctx, num_consumers, .consumers);
    demo.stats.setNum(demo.gctx, num_producers, .producers);
    Consumer.generateBulk(
        demo.gctx,
        demo.buffers.data.consumers.buf,
        &demo.buffers.data.consumers.mapping.num_structs,
        demo.aspect,
        num_consumers,
    );
    Producer.generateBulk(
        demo.gctx,
        demo.buffers.data.producers.buf,
        &demo.buffers.data.producers.mapping.num_structs,
        demo.aspect,
        num_producers,
        demo.sliders.get("production_rate").?.slider.val,
        demo.sliders.get("max_inventory").?.slider.val,
    );
    demo.stats.clear();
    demo.push_restart = false;
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
    const consumer_waiting = demo.buffers.data.consumers.mapping.state != .rest;
    const producer_waiting = demo.buffers.data.producers.mapping.state != .rest;
    if (consumer_waiting or producer_waiting) {
        demo.push_coord_update = true;
        return;
    }
    Wgpu.getAllAsync(Consumer, Callbacks.updateConsumerCoords, .{
        .gctx = demo.gctx,
        .obj_buf = &demo.buffers.data.consumers,
    });
    Wgpu.getAllAsync(Producer, Callbacks.updateProducerCoords, .{
        .gctx = demo.gctx,
        .obj_buf = &demo.buffers.data.producers,
    });

    demo.push_coord_update = false;
    demo.aspect = Camera.getAspectRatio(demo.gctx);
}

pub fn setImguiContentScale(scale: f32) void {
    zgui.getStyle().* = zgui.Style.init();
    zgui.getStyle().scaleAllSizes(scale);
    zgui.plot.getStyle().plot_padding = .{ 20 * scale, 20 * scale };
    zgui.plot.getStyle().line_weight = 2 * scale;
}

pub fn getContentScale(window: *zglfw.Window) f32 {
    const content_scale = window.getContentScale();
    if (emscripten) return 1;
    return @max(content_scale[0], content_scale[1]);
}
