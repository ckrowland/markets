const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const zmath = @import("zmath");
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Camera = @import("camera.zig");
const Circle = @import("circle.zig");
const Consumer = @import("consumer.zig");
const ConsumerHover = @import("consumer_hover.zig");
const Hover = @import("hover.zig");
const Producer = @import("producer.zig");
const Statistics = @import("statistics.zig");
const Wgpu = @import("wgpu.zig");
const Window = @import("windows.zig");
const Mouse = @import("mouse.zig");
const Popups = @import("popups.zig");
const Callbacks = @import("callbacks.zig");

pub const State = struct {
    pub const Selection = enum {
        none,
        consumer,
        consumers,
        producer,
    };
    selection: Selection = .consumer,
    producer: zgpu.TextureViewHandle = undefined,
    consumer: zgpu.TextureViewHandle = undefined,
    consumers: zgpu.TextureViewHandle = undefined,
    consumer_grouping_id: u32 = 0,
};

pub fn update(demo: *DemoState, selection_gui: *const fn () void) void {
    const gctx = demo.gctx;
    Window.setNextWindow(gctx, Window.ParametersWindow);
    if (zgui.begin("Parameters", Window.window_flags)) {
        zgui.pushIntId(2);
        selection_gui();
        parameters(demo, gctx);
        zgui.popId();
    }
    zgui.end();

    Window.setNextWindow(gctx, Window.StatsWindow);
    if (zgui.begin("Data", Window.window_flags)) {
        zgui.pushIntId(3);
        plots(demo);
        zgui.popId();
    }
    zgui.end();

    Wgpu.runCallbackIfReady(u32, &demo.buffers.data.stats.mapping);
    Wgpu.runCallbackIfReady(Producer, &demo.buffers.data.producers.mapping);
    Wgpu.runCallbackIfReady(Consumer, &demo.buffers.data.consumers.mapping);
    Wgpu.runCallbackIfReady(ConsumerHover, &demo.buffers.data.consumer_hovers.mapping);

    demo.popups.display(gctx, .{
        .consumers = &demo.buffers.data.consumers,
        .consumer_hovers = &demo.buffers.data.consumer_hovers,
        .mouse = demo.mouse,
        .producers = demo.buffers.data.producers,
        .stats = demo.buffers.data.stats,
        .allocator = demo.allocator,
    });

    hoverUpdate(gctx, demo);
    if (!demo.popups.anyOpen() and Mouse.onGrid(demo)) {
        _ = switch (demo.gui.selection) {
            .none => {},
            .consumer => {
                addingConsumer(gctx, demo, addConsumer);
            },
            .consumers => {
                addingConsumer(gctx, demo, addConsumerBrush);
            },
            .producer => {
                addingProducer(gctx, demo);
            },
        };
    }

    if (demo.running) {
        //Helpful for debugging shaders
        //Statistics.generateAndFillRandomColor(gctx, demo.buffers.data.stats.data);

        const current_time = @as(f32, @floatCast(gctx.stats.time));
        const seconds_passed = current_time - demo.stats.second;
        if (seconds_passed >= 1) {
            demo.stats.second = current_time;
            Wgpu.getAllAsync(u32, Callbacks.numTransactions, .{
                .gctx = demo.gctx,
                .buf = &demo.buffers.data.stats,
                .stats = &demo.stats,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.emptyConsumers, .{
                .gctx = demo.gctx,
                .buf = &demo.buffers.data.consumers,
                .stats = &demo.stats,
            });
            Wgpu.getAllAsync(Producer, Callbacks.totalInventory, .{
                .gctx = demo.gctx,
                .buf = &demo.buffers.data.producers,
                .stats = &demo.stats,
            });
        }
    }
}

fn hoverUpdate(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(demo.buffers.data.hover).?,
        @offsetOf(Hover, "position"),
        [2]f32,
        &.{demo.mouse.world_pos},
    );
}

fn addingConsumer(
    gctx: *zgpu.GraphicsContext,
    demo: *DemoState,
    addFn: *const fn (
        gctx: *zgpu.GraphicsContext,
        demo: *DemoState,
    ) void,
) void {
    const space_taken = demo.popups.doesAgentExist(demo.mouse.grid_pos);
    if (demo.mouse.down() and !space_taken) {
        addFn(gctx, demo);
    } else if (demo.mouse.released()) {
        const items = demo.buffers.data.consumers.list.items;
        const last_consumer = items[items.len - 1];
        demo.popups.appendPopup(.{
            .grid_center = last_consumer.absolute_home[0..2].*,
            .type_popup = .consumers,
            .parameters = .{
                .consumer = .{
                    .demand_rate = Consumer.defaults.demand_rate,
                    .moving_rate = Consumer.defaults.moving_rate,
                },
            },
        });
    }
}

fn addConsumer(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const gui_id = @as(u32, @intCast(demo.popups.popups.items.len));
    const consumer_args = Consumer.Args{
        .absolute_home = Camera.getGridFromWorld(gctx, demo.mouse.world_pos),
        .home = demo.mouse.world_pos,
        .grouping_id = gui_id,
    };

    Consumer.createAndAppend(gctx, .{
        .consumer = consumer_args,
        .obj_buf = &demo.buffers.data.consumers,
    });
    demo.buffers.data.consumers.mapping.staging.num_structs += 1;

    Statistics.setNum(gctx, .{
        .stat_obj = demo.buffers.data.stats,
        .num = @as(u32, @intCast(demo.buffers.data.consumers.list.items.len)),
        .param = .consumers,
    });

    ConsumerHover.createAndAppend(gctx, .{
        .args = consumer_args,
        .buf = &demo.buffers.data.consumer_hovers,
    });
    demo.buffers.data.consumer_hovers.mapping.staging.num_structs += 1;
    demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos);
}

fn addConsumerBrush(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const world_pos = demo.mouse.world_pos;
    const offset = 20;
    const array_positions: [5][2]f32 = .{
        world_pos,
        .{ world_pos[0] + offset, world_pos[1] + offset },
        .{ world_pos[0] - offset, world_pos[1] + offset },
        .{ world_pos[0] - offset, world_pos[1] - offset },
        .{ world_pos[0] + offset, world_pos[1] - offset },
    };
    const gui_id = @as(u32, @intCast(demo.popups.popups.items.len));
    for (array_positions) |pos| {
        Consumer.createAndAppend(gctx, .{
            .consumer = .{
                .absolute_home = Camera.getGridFromWorld(gctx, pos),
                .home = pos,
                .grouping_id = gui_id,
            },
            .obj_buf = &demo.buffers.data.consumers,
        });
        demo.buffers.data.consumers.mapping.staging.num_structs += 1;
    }
    Statistics.setNum(gctx, .{
        .stat_obj = demo.buffers.data.stats,
        .num = @as(u32, @intCast(demo.buffers.data.consumers.list.items.len)),
        .param = .consumers,
    });
    ConsumerHover.createAndAppend(gctx, .{
        .args = .{
            .absolute_home = Camera.getGridFromWorld(gctx, world_pos),
            .home = world_pos,
            .grouping_id = gui_id,
        },
        .buf = &demo.buffers.data.consumer_hovers,
    });
    demo.buffers.data.consumer_hovers.mapping.staging.num_structs += 1;
    demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos);
}

fn addingProducer(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const space_taken = demo.popups.doesAgentExist(demo.mouse.grid_pos);
    if (demo.mouse.pressed() and !space_taken) {
        Producer.createAndAppend(gctx, .{
            .producer = .{
                .home = demo.mouse.world_pos,
                .absolute_home = demo.mouse.grid_pos,
            },
            .obj_buf = &demo.buffers.data.producers,
        });
        Statistics.setNum(gctx, .{
            .stat_obj = demo.buffers.data.stats,
            .num = @as(u32, @intCast(demo.buffers.data.producers.list.items.len)),
            .param = .producers,
        });

        demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos);
        demo.popups.appendPopup(.{
            .grid_center = demo.mouse.grid_pos,
            .type_popup = .producer,
            .parameters = .{
                .producer = .{
                    .production_rate = Producer.DEFAULT_PRODUCTION_RATE,
                    .max_inventory = Producer.DEFAULT_MAX_INVENTORY,
                },
            },
        });
    }
}

fn plots(demo: *DemoState) void {
    const font_size = zgui.getFontSize() * 0.8;
    zgui.plot.pushStyleVar2f(.{
        .idx = .plot_padding,
        .v = .{ font_size, font_size },
    });

    const window_size = zgui.getWindowSize();
    const margin = 15;
    const plot_width = window_size[0] - margin;
    const plot_height = window_size[1] - margin;

    if (zgui.plot.beginPlot("", .{ .w = plot_width, .h = plot_height, .flags = .{} })) {
        zgui.plot.setupAxis(.x1, .{ .label = "", .flags = .{ .auto_fit = true } });
        zgui.plot.setupAxis(.y1, .{ .label = "", .flags = .{ .auto_fit = true } });
        zgui.plot.setupLegend(.{ .north = true, .west = true }, .{});
        zgui.plot.plotLineValues("Transactions", u32, .{
            .v = demo.stats.num_transactions.items[0..],
        });
        zgui.plot.plotLineValues("Empty Consumers", u32, .{
            .v = demo.stats.num_empty_consumers.items[0..],
        });
        zgui.plot.plotLineValues("Total Producer Inventory", u32, .{
            .v = demo.stats.num_total_producer_inventory.items[0..],
        });
        zgui.plot.endPlot();
    }
}

fn parameters(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
    const pressedColor = [4]f32{ 0.0, 0.5, 1.0, 1.0 };
    const size = 3 * zgui.getFontSize();
    const buttonSize = [2]f32{ size, size };

    zgui.text("Consumers", .{});
    if (demo.gui.selection == .consumers) {
        coloredButton(
            gctx,
            &demo.gui,
            State.Selection.consumers,
            demo.gui.consumers,
            pressedColor,
            buttonSize,
        );
    } else {
        coloredButton(
            gctx,
            &demo.gui,
            State.Selection.consumer,
            demo.gui.consumer,
            pressedColor,
            buttonSize,
        );
    }

    zgui.sameLine(.{});
    if (zgui.arrowButton("left_button_id", .{ .dir = .left })) {
        demo.gui.selection = .consumer;
    }

    zgui.sameLine(.{});
    if (zgui.arrowButton("right_button_id", .{ .dir = .right })) {
        demo.gui.selection = .consumers;
    }
    zgui.dummy(.{ .w = 1, .h = 10 });

    zgui.text("Producers", .{});
    coloredButton(
        gctx,
        &demo.gui,
        State.Selection.producer,
        demo.gui.producer,
        pressedColor,
        buttonSize,
    );
    zgui.dummy(.{ .w = 1, .h = 10 });

    zgui.text("Consumer Size", .{});
    if (zgui.sliderScalar("##cs", f32, .{ .v = &demo.params.consumer_radius, .min = 1, .max = 40 })) {
        demo.buffers.vertex.circle = Circle.createVertexBuffer(
            gctx,
            Main.NUM_CONSUMER_SIDES,
            demo.params.consumer_radius,
        );
    }

    if (zgui.button("Start", .{})) {
        demo.running = true;
    }

    if (zgui.button("Stop", .{})) {
        demo.running = false;
    }

    if (zgui.button("Clear", .{})) {
        demo.running = true;
        Main.clearSimulation(demo);
    }

    if (zgui.button("Supply Shock", .{})) {
        for (demo.buffers.data.producers.list.items, 0..) |_, i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.producers.buf).?,
                i * @sizeOf(Producer) + @offsetOf(Producer, "inventory"),
                i32,
                &.{0},
            );
        }
    }

    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("Set all producer inventory to 0.");
        zgui.endTooltip();
    }
}

fn coloredButton(
    gctx: *zgpu.GraphicsContext,
    guiState: *State,
    buttonState: State.Selection,
    textureView: zgpu.TextureViewHandle,
    color: [4]f32,
    size: [2]f32,
) void {
    const tex_id = gctx.lookupResource(textureView).?;
    const id = @tagName(buttonState);
    const pixel_size = .{
        .w = size[0],
        .h = size[1],
    };
    if (guiState.selection == buttonState) {
        zgui.pushStyleColor4f(.{ .idx = .button, .c = color });
        defer zgui.popStyleColor(.{});
        if (zgui.imageButton(id, tex_id, pixel_size)) {
            guiState.selection = .none;
        }
    } else {
        if (zgui.imageButton(id, tex_id, pixel_size)) {
            guiState.selection = buttonState;
        }
    }
}
