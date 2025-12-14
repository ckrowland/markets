const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const zgui = @import("zgui");
const zm = @import("zmath");
const Callbacks = @import("callbacks.zig");
const gui = @import("gui.zig");
const Sliders = @import("sliders.zig");
const Wgpu = @import("libs/wgpu.zig");
const Consumer = @import("libs/consumer.zig");
const Producer = @import("libs/producer.zig");
const Camera = @import("libs/camera.zig");
const Shapes = @import("libs/shapes.zig");

pub const MAX_NUM_PRODUCERS = 1000;
pub const MAX_NUM_CONSUMERS = 10000;
pub const NUM_STATS = 12;
pub const NUM_CONSUMER_SIDES = 20;

pub const Self = @This();
gctx: *zgpu.GraphicsContext,
window: *zglfw.Window,
imgui_windows: struct {
    sliders: gui.Window = .{
        .pos = .{ .x = 0, .y = 0 },
        .size = .{ .x = 0.2, .y = 1 },
        .window_fn = &gui.settings,
        .window_flags = .{
            .no_move = true,
            .no_title_bar = true,
            .no_resize = true,
            .no_scrollbar = true,
            .no_collapse = true,
        },
    },
    statistics: gui.Window = .{
        .pos = .{ .x = 0.2, .y = 0, .margin = .{ .left = false } },
        .size = .{ .x = 0.8, .y = 1, .margin = .{ .right = false } },
        .window_fn = &gui.plots,
        .p_open = false,
    },
    help: gui.Window = .{
        .pos = .{ .x = 0.2, .y = 0, .margin = .{ .left = false } },
        .size = .{ .x = 0.8, .y = 1, .margin = .{ .right = false } },
        .window_fn = &gui.help,
        .p_open = false,
    },
} = .{},
allocator: std.mem.Allocator,
running: bool = false,
space_press: bool = false,
restart_press: bool = false,
content_scale: f32,
compute_pipelines: struct {
    consumer: zgpu.ComputePipelineHandle,
    producer: zgpu.ComputePipelineHandle,
},
bind_groups: struct {
    object_to_clip: zgpu.BindGroupHandle,
    compute: zgpu.BindGroupHandle,
    price: zgpu.BindGroupHandle,
},
graphics_objects: struct {
    consumers: Wgpu.GraphicsObject,
    consumers_money: Wgpu.GraphicsObject,
    producers: Wgpu.GraphicsObject,
    producers_money: Wgpu.GraphicsObject,
    producers_bar: Wgpu.GraphicsObject,
    producers_tick: Wgpu.GraphicsObject,
},
buffers: struct {
    consumers: Wgpu.ObjectBuffer(Consumer),
    producers: Wgpu.ObjectBuffer(Producer),
},
depth_texture: zgpu.TextureHandle,
depth_texture_view: zgpu.TextureViewHandle,
params: Sliders.Parameters,
stats: struct {
    num_transactions: std.ArrayList(u32),
    price: std.ArrayList(u32),
    second: f32 = 0,
    num_empty_consumers: std.ArrayList(u32),
    avg_producer_inventory: std.ArrayList(u32),
    avg_producer_money: std.ArrayList(u32),
    avg_consumer_inventory: std.ArrayList(u32),
    avg_consumer_money: std.ArrayList(u32),
    obj_buf: Wgpu.ObjectBuffer(u32),
},

pub fn updateAndRender(demo: *Self) !void {
    zglfw.pollEvents();
    switch (zglfw.getKey(demo.window, .caps_lock)) {
        .release => demo.restart_press = false,
        .press => {
            if (!demo.restart_press) {
                restartSimulation(demo);
                demo.restart_press = true;
            }
        },
        else => {},
    }
    switch (zglfw.getKey(demo.window, .space)) {
        .release => demo.space_press = false,
        .press => {
            if (!demo.space_press) {
                demo.running = !demo.running;
                demo.space_press = true;
            }
        },
        else => {},
    }

    zgui.backend.newFrame(
        demo.gctx.swapchain_descriptor.width,
        demo.gctx.swapchain_descriptor.height,
    );
    gui.update(demo);
    draw(demo);
    demo.window.swapBuffers();
}

pub fn getContentScale(window: *zglfw.Window) f32 {
    const content_scale = window.getContentScale();
    return @max(content_scale[0], content_scale[1]);
}

pub fn deinit(demo: *Self) void {
    demo.stats.avg_producer_money.deinit();
    demo.stats.avg_producer_inventory.deinit();
    demo.stats.avg_consumer_money.deinit();
    demo.stats.avg_consumer_inventory.deinit();
    demo.stats.num_transactions.deinit();
    demo.stats.num_empty_consumers.deinit();
    demo.stats.price.deinit();
    zgui.backend.deinit();
    zgui.plot.deinit();
    zgui.deinit();
    demo.gctx.destroy(demo.allocator);
    demo.window.destroy();
    zglfw.terminate();
}

pub fn init(allocator: std.mem.Allocator) !Self {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);

    var window = try zglfw.Window.create(1600, 900, "Simulations", null);
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

    const dark_grey = .{ 0.2, 0.2, 0.2, 1 };
    const medium_grey = .{ 0.4, 0.4, 0.4, 1 };
    const light_grey = .{ 0.6, 0.6, 0.6, 1 };
    const white = .{ 1, 1, 1, 1 };

    zgui.getStyle().*.setColor(.separator, dark_grey);
    zgui.getStyle().*.setColor(.separator_active, light_grey);
    zgui.getStyle().*.setColor(.separator_hovered, medium_grey);

    zgui.getStyle().*.setColor(.resize_grip, dark_grey);
    zgui.getStyle().*.setColor(.resize_grip_active, dark_grey);
    zgui.getStyle().*.setColor(.resize_grip_hovered, dark_grey);

    zgui.getStyle().*.setColor(.frame_bg, dark_grey);
    zgui.getStyle().*.setColor(.frame_bg_active, medium_grey);
    zgui.getStyle().*.setColor(.frame_bg_hovered, medium_grey);

    zgui.getStyle().*.setColor(.slider_grab, light_grey);
    zgui.getStyle().*.setColor(.slider_grab_active, light_grey);

    zgui.getStyle().*.setColor(.tab, dark_grey);
    zgui.getStyle().*.setColor(.tab_hovered, medium_grey);
    zgui.getStyle().*.setColor(.tab_selected, medium_grey);

    zgui.getStyle().*.setColor(.button, dark_grey);
    zgui.getStyle().*.setColor(.button_hovered, medium_grey);
    zgui.getStyle().*.setColor(.button_active, light_grey);

    zgui.getStyle().*.setColor(.title_bg, dark_grey);
    zgui.getStyle().*.setColor(.title_bg_active, medium_grey);
    zgui.getStyle().*.setColor(.title_bg_collapsed, dark_grey);

    zgui.plot.getStyle().*.setColor(.line, white);

    var params = Sliders.Parameters{ .aspect = Camera.getAspectRatio(gctx) };
    Camera.updateMaxX(gctx);

    var consumer_object = Wgpu.createObjectBuffer(gctx, Consumer, MAX_NUM_CONSUMERS, 0);
    Consumer.generateBulk(
        gctx,
        &consumer_object,
        params.num_consumers.val,
        .{
            .income = params.income.val,
            .moving_rate = params.moving_rate.val,
            .max_money = params.max_consumer_money.val,
        },
    );
    params.num_consumers.val += 3;

    var producer_object = Wgpu.createObjectBuffer(gctx, Producer, MAX_NUM_PRODUCERS, 0);
    Producer.generateBulk(
        gctx,
        &producer_object,
        params.num_producers.val,
        .{
            .max_inventory = params.max_inventory.val,
            .production_cost = params.production_cost.val,
            .max_production_rate = params.max_production_rate.val,
            .price = params.price.val,
            .max_money = params.max_producer_money.val,
            .decay_rate = params.decay_rate.val,
        },
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
        &.{ params.num_consumers.val, params.num_producers.val },
    );
    gctx.queue.writeBuffer(
        resource,
        4 * @sizeOf(u32),
        u32,
        &.{ Camera.MAX_X, Camera.MAX_Y },
    );

    const compute_bind_group = Wgpu.createComputeBindGroup(gctx, .{
        .consumer = consumer_object.buf,
        .producer = producer_object.buf,
        .stats = stats_buf.buf,
    });
    const depth = Wgpu.createDepthTexture(gctx);

    const bgl = gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
    });
    const o2c_bg = gctx.createBindGroup(bgl, &.{
        .{
            .binding = 0,
            .buffer_handle = gctx.uniforms.buffer,
            .offset = 0,
            .size = @sizeOf(zm.Mat),
        },
    });

    const price_bg = gctx.createBindGroup(bgl, &.{
        .{
            .binding = 0,
            .buffer_handle = gctx.uniforms.buffer,
            .offset = 0,
            .size = @sizeOf(u32),
        },
    });

    return Self{
        .gctx = gctx,
        .window = window,
        .content_scale = getContentScale(window),
        .compute_pipelines = .{
            .producer = Wgpu.createComputePipeline(gctx, .{
                .cs = @embedFile("shaders/compute/common.wgsl") ++
                    @embedFile("shaders/compute/producer.wgsl"),
                .entry_point = "main",
                .name = "producer",
                .constants = .{
                    .max_num_consumers = MAX_NUM_CONSUMERS,
                    .max_num_producers = MAX_NUM_PRODUCERS,
                },
            }),
            .consumer = Wgpu.createComputePipeline(gctx, .{
                .cs = @embedFile("shaders/compute/common.wgsl") ++
                    @embedFile("shaders/compute/consumer.wgsl"),
                .entry_point = "main",
                .name = "consumer",
                .constants = .{
                    .max_num_consumers = MAX_NUM_CONSUMERS,
                    .max_num_producers = MAX_NUM_PRODUCERS,
                },
            }),
        },
        .bind_groups = .{
            .object_to_clip = o2c_bg,
            .compute = compute_bind_group,
            .price = price_bg,
        },
        .graphics_objects = .{
            .consumers_money = .{
                .render_pipeline = Wgpu.createRenderPipeline(
                    gctx,
                    &.{ bgl, bgl },
                    .{
                        .vs = @embedFile("shaders/vertex/consumer/money_circle.wgsl"),
                        .inst_type = Consumer,
                        .inst_attrs = &.{
                            .{ .name = "home", .type = [4]f32 },
                            .{ .name = "money", .type = u32 },
                        },
                        .primitive_topology = .line_list,
                    },
                ),
                .attribute_buffer = consumer_object.buf,
                .vertex_buffer = Shapes.createCircleVertexBuffer(
                    gctx,
                    NUM_CONSUMER_SIDES,
                    params.consumer_size.val,
                ),
                .index_buffer = Shapes.createMoneyCircleIndexBuffer(gctx, NUM_CONSUMER_SIDES),
            },
            .consumers = .{
                .render_pipeline = Wgpu.createRenderPipeline(
                    gctx,
                    &.{bgl},
                    .{
                        .vs = @embedFile("shaders/vertex/consumer/circle.wgsl"),
                        .inst_type = Consumer,
                        .inst_attrs = &.{
                            .{ .name = "position", .type = [4]f32 },
                            .{ .name = "color", .type = [4]f32 },
                            .{ .name = "inventory", .type = u32 },
                        },
                    },
                ),
                .attribute_buffer = consumer_object.buf,
                .vertex_buffer = Shapes.createCircleVertexBuffer(
                    gctx,
                    NUM_CONSUMER_SIDES,
                    params.consumer_size.val,
                ),
                .index_buffer = Shapes.createCircleIndexBuffer(gctx, NUM_CONSUMER_SIDES),
            },
            .producers = .{
                .render_pipeline = Wgpu.createRenderPipeline(gctx, &.{bgl}, .{
                    .vs = @embedFile("shaders/vertex/producer/square.wgsl"),
                    .inst_type = Producer,
                    .inst_attrs = &.{
                        .{ .name = "home", .type = [4]f32 },
                        .{ .name = "color", .type = [4]f32 },
                        .{ .name = "inventory", .type = u32 },
                    },
                }),
                .attribute_buffer = producer_object.buf,
                .vertex_buffer = Shapes.createRectangleVertexBuffer(
                    gctx,
                    params.producer_size.val,
                    params.producer_size.val,
                ),
                .index_buffer = Shapes.createRectangleIndexBuffer(gctx),
            },
            .producers_money = .{
                .render_pipeline = Wgpu.createRenderPipeline(gctx, &.{ bgl, bgl }, .{
                    .vs = @embedFile("shaders/vertex/producer/money_square.wgsl"),
                    .inst_type = Producer,
                    .inst_attrs = &.{
                        .{ .name = "home", .type = [4]f32 },
                        .{ .name = "money", .type = u32 },
                    },
                    .primitive_topology = .line_list,
                }),
                .attribute_buffer = producer_object.buf,
                .vertex_buffer = Shapes.createRectangleVertexBuffer(
                    gctx,
                    params.producer_size.val,
                    params.producer_size.val,
                ),
                .index_buffer = Shapes.createMoneySquareIndexBuffer(gctx),
            },
            .producers_bar = .{
                .render_pipeline = Wgpu.createRenderPipeline(gctx, &.{bgl}, .{
                    .vs = @embedFile("shaders/vertex/producer/price_bar.wgsl"),
                    .inst_type = Producer,
                    .inst_attrs = &.{
                        .{ .name = "home", .type = [4]f32 },
                        .{ .name = "color", .type = [4]f32 },
                    },
                }),
                .attribute_buffer = producer_object.buf,
                .vertex_buffer = Shapes.createRectangleVertexBuffer(
                    gctx,
                    params.producer_size.val / 2,
                    params.producer_size.val * 8,
                ),
                .index_buffer = Shapes.createRectangleIndexBuffer(gctx),
            },
            .producers_tick = .{
                .render_pipeline = Wgpu.createRenderPipeline(gctx, &.{bgl}, .{
                    .vs = @embedFile("shaders/vertex/producer/price_tick.wgsl"),
                    .inst_type = Producer,
                    .inst_attrs = &.{
                        .{ .name = "home", .type = [4]f32 },
                        .{ .name = "color", .type = [4]f32 },
                        .{ .name = "price", .type = u32 },
                    },
                }),
                .attribute_buffer = producer_object.buf,
                .vertex_buffer = Shapes.createRectangleVertexBuffer(
                    gctx,
                    params.producer_size.val,
                    params.producer_size.val / 2,
                ),
                .index_buffer = Shapes.createRectangleIndexBuffer(gctx),
            },
        },
        .buffers = .{
            .consumers = consumer_object,
            .producers = producer_object,
        },
        .depth_texture = depth.texture,
        .depth_texture_view = depth.view,
        .allocator = allocator,
        .params = params,
        .stats = .{
            .avg_producer_money = std.ArrayList(u32).init(allocator),
            .avg_producer_inventory = std.ArrayList(u32).init(allocator),
            .avg_consumer_money = std.ArrayList(u32).init(allocator),
            .avg_consumer_inventory = std.ArrayList(u32).init(allocator),
            .num_transactions = std.ArrayList(u32).init(allocator),
            .num_empty_consumers = std.ArrayList(u32).init(allocator),
            .obj_buf = stats_buf,
            .price = std.ArrayList(u32).init(allocator),
        },
    };
}

pub fn draw(demo: *Self) void {
    const gctx = demo.gctx;
    const cam_world_to_clip = Camera.getObjectToClipMat();

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        const num_consumers = demo.buffers.consumers.mapping.num_structs;
        const num_producers = demo.buffers.producers.mapping.num_structs;

        const sd = gctx.swapchain_descriptor;
        const sd_width = @as(f32, @floatFromInt(sd.width));
        const x_offset = sd_width * (1 - Camera.VP_X_SIZE);
        const sd_height = @as(f32, @floatFromInt(sd.height));
        const y_offset = sd_height * (1 - Camera.VP_Y_SIZE);
        const width = sd_width - x_offset;
        const height = sd_height - y_offset;

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
                pass.setPipeline(ccp);
                pass.dispatchWorkgroups(@divFloor(num_consumers, 64) + 1, 1, 1);
                pass.setPipeline(pcp);
                pass.dispatchWorkgroups(@divFloor(num_producers, 64) + 1, 1, 1);
            }
        }

        const stat_buf = demo.stats.obj_buf.mapping;
        if (stat_buf.insert_idx != stat_buf.remove_idx) {
            pass: {
                const s = gctx.lookupResource(demo.stats.obj_buf.buf) orelse break :pass;
                const s_info = gctx.lookupResourceInfo(demo.stats.obj_buf.buf) orelse break :pass;
                const sm = gctx.lookupResource(demo.stats.obj_buf.mapping.buf) orelse break :pass;
                encoder.copyBufferToBuffer(s, 0, sm, 0, s_info.size);
                demo.stats.obj_buf.mapping.state = .call_map_async;
            }
        }

        const p_buf = demo.buffers.producers.mapping;
        if (p_buf.insert_idx != p_buf.remove_idx) {
            pass: {
                const p = gctx.lookupResource(demo.buffers.producers.buf) orelse break :pass;
                const p_info = gctx.lookupResourceInfo(demo.buffers.producers.buf) orelse break :pass;
                const pm = gctx.lookupResource(demo.buffers.producers.mapping.buf) orelse break :pass;
                encoder.copyBufferToBuffer(p, 0, pm, 0, p_info.size);
                demo.buffers.producers.mapping.state = .call_map_async;
            }
        }

        const c_buf = demo.buffers.consumers.mapping;
        if (c_buf.insert_idx != c_buf.remove_idx) {
            pass: {
                const c = gctx.lookupResource(demo.buffers.consumers.buf) orelse break :pass;
                const c_info = gctx.lookupResourceInfo(demo.buffers.consumers.buf) orelse break :pass;
                const cm = gctx.lookupResource(demo.buffers.consumers.mapping.buf) orelse break :pass;
                encoder.copyBufferToBuffer(c, 0, cm, 0, c_info.size);
                demo.buffers.consumers.mapping.state = .call_map_async;
            }
        }

        pass: {
            const o2c_bg = gctx.lookupResource(demo.bind_groups.object_to_clip) orelse break :pass;
            const price_bg = gctx.lookupResource(demo.bind_groups.price) orelse break :pass;
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
            pass.setViewport(x_offset, y_offset / 2, width, height, 0, 1);

            var mem = gctx.uniformsAllocate(zm.Mat, 1);
            mem.slice[0] = cam_world_to_clip;
            pass.setBindGroup(0, o2c_bg, &.{mem.offset});

            var price = gctx.uniformsAllocate(u32, 1);
            price.slice[0] = demo.params.price.val;
            pass.setBindGroup(1, price_bg, &.{price.offset});

            demo.graphics_objects.consumers_money.renderGraphicsObject(gctx, pass, num_consumers);
            demo.graphics_objects.consumers.renderGraphicsObject(gctx, pass, num_consumers);
            demo.graphics_objects.producers.renderGraphicsObject(gctx, pass, num_producers);
            demo.graphics_objects.producers_bar.renderGraphicsObject(gctx, pass, num_producers);
            demo.graphics_objects.producers_tick.renderGraphicsObject(gctx, pass, num_producers);

            var pc = gctx.uniformsAllocate(u32, 1);
            pc.slice[0] = demo.params.production_cost.val;
            pass.setBindGroup(1, price_bg, &.{pc.offset});
            demo.graphics_objects.producers_money.renderGraphicsObject(gctx, pass, num_producers);
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
        zgui.getStyle().scaleAllSizes(demo.content_scale);
        updateAspectRatio(demo);
    }
}

pub fn restartSimulation(demo: *Self) void {
    const encoder = demo.gctx.device.createCommandEncoder(null);
    defer encoder.release();

    Wgpu.clearObjBuffer(encoder, demo.gctx, Consumer, &demo.buffers.consumers);
    Wgpu.clearObjBuffer(encoder, demo.gctx, Producer, &demo.buffers.producers);
    Wgpu.clearObjBuffer(encoder, demo.gctx, u32, &demo.stats.obj_buf);
    demo.stats.obj_buf.mapping.num_structs = NUM_STATS;

    demo.params.price.val = demo.params.price.restart.?;
    demo.params.num_producers.val = demo.params.num_producers.restart.?;
    demo.params.num_consumers.val = demo.params.num_consumers.restart.?;
    const resource = demo.gctx.lookupResource(demo.stats.obj_buf.buf).?;
    demo.gctx.queue.writeBuffer(
        resource,
        @sizeOf(u32),
        u32,
        &.{ demo.params.num_consumers.val, demo.params.num_producers.val },
    );

    Consumer.generateBulk(
        demo.gctx,
        &demo.buffers.consumers,
        demo.params.num_consumers.val,
        .{
            .income = demo.params.income.val,
            .moving_rate = demo.params.moving_rate.val,
            .max_money = demo.params.max_consumer_money.val,
        },
    );
    Producer.generateBulk(
        demo.gctx,
        &demo.buffers.producers,
        demo.params.num_producers.val,
        .{
            .max_inventory = demo.params.max_inventory.val,
            .production_cost = demo.params.production_cost.val,
            .max_production_rate = demo.params.max_production_rate.val,
            .price = demo.params.price.val,
            .max_money = demo.params.max_producer_money.val,
            .decay_rate = demo.params.decay_rate.val,
        },
    );

    demo.stats.num_transactions.clearAndFree();
    demo.stats.num_empty_consumers.clearAndFree();
    demo.stats.avg_producer_inventory.clearAndFree();
    demo.stats.avg_producer_money.clearAndFree();
    demo.stats.avg_consumer_inventory.clearAndFree();
    demo.stats.avg_consumer_money.clearAndFree();
    demo.stats.price.clearAndFree();
}

pub fn updateDepthTexture(demo: *Self) void {
    // Release old depth texture.
    demo.gctx.releaseResource(demo.depth_texture_view);
    demo.gctx.destroyResource(demo.depth_texture);

    // Create a new depth texture to match the new window size.
    const depth = Wgpu.createDepthTexture(demo.gctx);
    demo.depth_texture = depth.texture;
    demo.depth_texture_view = depth.view;
}

pub fn updateAspectRatio(demo: *Self) void {
    updateDepthTexture(demo);
    Wgpu.getAllAsync(Consumer, Callbacks.updateConsumerCoords, .{
        .gctx = demo.gctx,
        .obj_buf = &demo.buffers.consumers,
    });
    Wgpu.getAllAsync(Producer, Callbacks.updateProducerCoords, .{
        .gctx = demo.gctx,
        .obj_buf = &demo.buffers.producers,
        .stat_buf = &demo.stats.obj_buf,
    });
    demo.params.aspect = Camera.getAspectRatio(demo.gctx);
    demo.imgui_windows.sliders.change = true;
    demo.imgui_windows.statistics.change = true;
    demo.imgui_windows.help.change = true;
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
