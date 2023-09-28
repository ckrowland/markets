const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const zmath = @import("zmath");

const Main = @import("main.zig");
const Camera = @import("../../camera.zig");
const Circle = @import("circle.zig");
const Consumer = @import("../consumer.zig");
const ConsumerHover = @import("consumer_hover.zig");
const DemoState = @import("main.zig");
const Hover = @import("hover.zig");
const Producer = @import("../producer.zig");
const Square = @import("square.zig");
const Statistics = @import("../statistics.zig");
const Wgpu = @import("../wgpu.zig");
const Window = @import("../../windows.zig");
const Mouse = @import("mouse.zig");
const Popups = @import("popups.zig");

pub const State = struct {
    pub const Selection = enum {
        none,
        consumer,
        consumers,
        producer,
    };
    selection: Selection = .producer,
    producer: zgpu.TextureViewHandle,
    consumer: zgpu.TextureViewHandle,
    consumers: zgpu.TextureViewHandle,
    consumer_grouping_id: u32 = 0,
};

pub fn update(demo: *DemoState, gctx: *zgpu.GraphicsContext) !void {
    Window.setNextWindow(gctx, Window.ParametersWindow);
    if (zgui.begin("Parameters", Window.window_flags)) {
        zgui.pushIntId(2);
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

    try demo.popups.display(gctx, .{
        .consumers = demo.buffers.data.consumer,
        .consumer_hover = demo.buffers.data.consumer_hover,
        .consumer_hover_len = demo.params.num_consumer_hovers,
        .mouse = demo.mouse,
        .producers = demo.buffers.data.producer,
        .stats = demo.buffers.data.stats,
        .allocator = demo.allocator,
    });

    if (demo.running) {
        Statistics.generateAndFillRandomColor(gctx, demo.buffers.data.stats.data);
        const current_time = @as(f32, @floatCast(gctx.stats.time));
        const seconds_passed = current_time - demo.stats.second;
        if (seconds_passed >= 1) {
            demo.stats.second = current_time;
            demo.stats.update(gctx, .{
                .stats = demo.buffers.data.stats,
                .consumers = demo.buffers.data.consumer,
                .producers = demo.buffers.data.producer,
            });
        }
    }

    hoverUpdate(gctx, demo);
    if (!demo.popups.anyOpen()) {
        _ = switch (demo.gui.selection) {
            .none => {},
            .consumer => {
                try addingConsumer(gctx, demo, addConsumer);
            },
            .consumers => {
                try addingConsumer(gctx, demo, addConsumerBrush);
            },
            .producer => {
                try addingProducer(gctx, demo);
            },
        };
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
    ) std.mem.Allocator.Error!void,
) !void {
    const space_taken = demo.popups.doesAgentExist(demo.mouse.grid_pos);
    const on_grid = Mouse.onGrid(gctx);
    if (demo.mouse.down() and !space_taken and on_grid) {
        try addFn(gctx, demo);
    } else if (demo.mouse.released()) {
        const prev_consumer = Wgpu.getLast(gctx, Consumer, .{
            .structs = demo.buffers.data.consumer,
            .num_structs = demo.params.num_consumers,
        }) catch return;
        try demo.popups.appendPopup(.{
            .grid_agent_center = prev_consumer.absolute_home[0..2].*,
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

fn addConsumer(gctx: *zgpu.GraphicsContext, demo: *DemoState) !void {
    const gui_id = @as(u32, @intCast(demo.popups.popups.items.len));
    const consumer_args = Consumer.Args{
        .absolute_home = Camera.getGridPosition(gctx, demo.mouse.world_pos),
        .home = demo.mouse.world_pos,
        .grouping_id = gui_id,
    };

    Consumer.createAndAppend(gctx, .{
        .consumer_args = consumer_args,
        .consumer_buf = demo.buffers.data.consumer.data,
        .num_consumers = demo.params.num_consumers,
    });
    demo.params.num_consumers += 1;
    Statistics.setNumConsumers(gctx, demo.buffers.data.stats, demo.params.num_consumers);

    ConsumerHover.createAndAppend(gctx, .{
        .hover_args = consumer_args,
        .hover_buf = demo.buffers.data.consumer_hover.data,
        .num_consumer_hovers = demo.params.num_consumer_hovers,
    });
    demo.params.num_consumer_hovers += 1;
    Statistics.setNumConsumerHovers(gctx, demo.buffers.data.stats, demo.params.num_consumer_hovers);

    demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos) catch unreachable;
}

fn addConsumerBrush(gctx: *zgpu.GraphicsContext, demo: *DemoState) !void {
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
            .consumer_args = .{
                .absolute_home = Camera.getGridPosition(gctx, pos),
                .home = pos,
                .grouping_id = gui_id,
            },
            .consumer_buf = demo.buffers.data.consumer.data,
            .num_consumers = demo.params.num_consumers,
        });
        demo.params.num_consumers += 1;
        Statistics.setNumConsumers(
            gctx,
            demo.buffers.data.stats,
            demo.params.num_consumers,
        );
    }
    ConsumerHover.createAndAppend(gctx, .{
        .hover_args = .{
            .absolute_home = Camera.getGridPosition(gctx, world_pos),
            .home = world_pos,
            .grouping_id = gui_id,
        },
        .hover_buf = demo.buffers.data.consumer_hover.data,
        .num_consumer_hovers = demo.params.num_consumer_hovers,
    });
    demo.params.num_consumer_hovers += 1;
    demo.popups.appendSquare(
        demo.allocator,
        demo.mouse.grid_pos,
    ) catch unreachable;
}

fn addingProducer(gctx: *zgpu.GraphicsContext, demo: *DemoState) !void {
    const space_taken = demo.popups.doesAgentExist(demo.mouse.grid_pos);
    const on_grid = Mouse.onGrid(gctx);

    if (demo.mouse.pressed() and on_grid and !space_taken) {
        var producer = Producer.create(.{
            .home = demo.mouse.world_pos,
            .absolute_home = demo.mouse.grid_pos,
        });
        var producers = [1]Producer{producer};
        Wgpu.appendBuffer(gctx, Producer, .{
            .num_old_structs = demo.params.num_producers,
            .buf = demo.buffers.data.producer.data,
            .structs = producers[0..],
        });
        demo.params.num_producers += 1;
        Statistics.setNumProducers(
            gctx,
            demo.buffers.data.stats,
            demo.params.num_producers,
        );

        try demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos);
        try demo.popups.appendPopup(.{
            .grid_agent_center = demo.mouse.grid_pos,
            .type_popup = .producer,
            .parameters = .{
                .producer = .{
                    .production_rate = producer.production_rate,
                    .max_inventory = producer.max_inventory,
                },
            },
        });
    }
}

fn plots(demo: *DemoState) void {
    const font_size = zgui.getFontSize();
    zgui.plot.pushStyleVar2f(.{
        .idx = .plot_padding,
        .v = .{ font_size, font_size },
    });

    const window_size = zgui.getWindowSize();
    if (zgui.plot.beginPlot("", .{ .w = window_size[0], .h = window_size[1], .flags = .{} })) {
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
    zgui.spacing();

    coloredButton(
        gctx,
        &demo.gui,
        State.Selection.producer,
        demo.gui.producer,
        pressedColor,
        buttonSize,
    );

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
        DemoState.restartSimulation(demo, gctx);
    }

    if (zgui.button("Supply Shock", .{})) {
        Wgpu.setAll(gctx, Producer, Wgpu.setArgs(Producer){
            .agents = demo.buffers.data.producer,
            .num_structs = demo.params.num_producers,
            .parameter = .{
                .inventory = 0,
            },
        });
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
