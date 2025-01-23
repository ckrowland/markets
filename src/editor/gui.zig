const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const zmath = @import("zmath");
const Shapes = @import("shapes");
const Camera = @import("camera");
const Gui = @import("gui");
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Consumer = @import("consumer");
const ConsumerHover = @import("consumer_hover.zig");
const Hover = @import("hover.zig");
const Producer = @import("producer");
const Statistics = @import("statistics");
const Wgpu = @import("wgpu");
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

pub fn update(demo: *DemoState) void {
    const gctx = demo.gctx;
    const sd = demo.gctx.swapchain_descriptor;
    Gui.setupWindowPos(sd, .{ .x = 0, .y = 0 });
    Gui.setupWindowSize(sd, .{ .x = 0.25, .y = 0.75 });
    const flags = zgui.WindowFlags.no_decoration;

    if (zgui.begin("Parameters", .{ .flags = flags })) {
        zgui.pushIntId(2);
        parameters(demo, gctx);
        zgui.popId();
    }
    zgui.end();

    const pos: Gui.Pos = .{ .x = 0, .y = 0.75, .margin = .{ .top = false } };
    const size: Gui.Pos = .{ .x = 1, .y = 0.25, .margin = .{ .top = false } };
    Gui.setupWindowPos(sd, pos);
    Gui.setupWindowSize(sd, size);
    if (zgui.begin("Data", .{ .flags = flags })) {
        zgui.pushIntId(3);
        plots(demo);
        zgui.popId();
    }
    zgui.end();

    Wgpu.checkObjBufState(u32, &demo.stats.obj_buf.mapping);
    Wgpu.checkObjBufState(Producer, &demo.buffers.data.producers.mapping);
    Wgpu.checkObjBufState(Consumer, &demo.buffers.data.consumers.mapping);
    Wgpu.checkObjBufState(ConsumerHover, &demo.buffers.data.consumer_hovers.mapping);

    demo.popups.display(gctx, .{
        .consumers = &demo.buffers.data.consumers,
        .consumer_params = demo.buffers.data.consumer_params,
        .consumer_hovers = &demo.buffers.data.consumer_hovers,
        .consumer_hover_colors = demo.buffers.data.consumer_hover_colors,
        .mouse = demo.mouse,
        .producers = demo.buffers.data.producers,
        .allocator = demo.allocator,
        .content_scale = demo.content_scale,
    });

    updateMouseHover(gctx, demo);

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
                .obj_buf = &demo.stats.obj_buf,
                .stat_array = &demo.stats.num_transactions,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.emptyConsumers, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.consumers,
                .stat_array = &demo.stats.num_empty_consumers,
            });
            Wgpu.getAllAsync(Producer, Callbacks.totalInventory, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.producers,
                .stat_array = &demo.stats.num_total_producer_inventory,
            });
        }
    }
}

fn updateMouseHover(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const resource = gctx.lookupResource(demo.buffers.data.hover).?;
    const offset = @offsetOf(Hover, "position");
    const new_pos = demo.mouse.world_pos[0..2].*;
    gctx.queue.writeBuffer(resource, offset, [2]f32, &.{new_pos});
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
        demo.popups.appendPopup(.{
            .grid_center = demo.mouse.grid_pos[0..2].*,
            .type_popup = .consumers,
            .parameters = .{
                .consumer = .{
                    .demand_rate = demo.params.demand_rate,
                    .moving_rate = demo.params.moving_rate,
                },
            },
        });

        const last = demo.popups.consumers_popups.getLast();
        const buf_offset = last.id.gui_id * 8;
        const r = gctx.lookupResource(demo.buffers.data.consumer_params).?;
        gctx.queue.writeBuffer(r, buf_offset, f32, &.{demo.params.moving_rate});
        gctx.queue.writeBuffer(r, buf_offset + 4, u32, &.{demo.params.demand_rate});
    }
}

fn addConsumer(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const gui_id = @as(u32, @intCast(demo.popups.consumers_popups.items.len));
    const consumer = Consumer{
        .absolute_home = Camera.getGridFromWorld(gctx, demo.mouse.world_pos),
        .home = demo.mouse.world_pos,
        .grouping_id = gui_id,
    };

    Consumer.createAndAppend(
        gctx,
        demo.buffers.data.consumers.buf,
        &demo.buffers.data.consumers.mapping.num_structs,
        consumer,
    );

    const num_consumers = demo.buffers.data.consumers.mapping.num_structs;
    demo.stats.setNum(gctx, num_consumers, .consumers);

    ConsumerHover.createAndAppend(demo, consumer);
    demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos, .consumers);
}

fn addConsumerBrush(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const world_pos = demo.mouse.world_pos;
    const offset = 20;
    const array_positions: [5][4]f32 = .{
        .{ world_pos[0] + offset, world_pos[1] + offset, 0, 1 },
        .{ world_pos[0] - offset, world_pos[1] + offset, 0, 1 },
        .{ world_pos[0] - offset, world_pos[1] - offset, 0, 1 },
        .{ world_pos[0] + offset, world_pos[1] - offset, 0, 1 },
        world_pos,
    };
    const gui_id = @as(u32, @intCast(demo.popups.consumers_popups.items.len));
    var consumer: Consumer = undefined;
    for (array_positions) |pos| {
        consumer = Consumer{
            .absolute_home = Camera.getGridFromWorld(gctx, pos),
            .home = pos,
            .grouping_id = gui_id,
        };
        Consumer.createAndAppend(
            gctx,
            demo.buffers.data.consumers.buf,
            &demo.buffers.data.consumers.mapping.num_structs,
            consumer,
        );
    }
    const num_consumers = demo.buffers.data.consumers.mapping.num_structs;
    demo.stats.setNum(gctx, num_consumers, .consumers);

    ConsumerHover.createAndAppend(demo, consumer);
    demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos, .consumers);
}

fn addingProducer(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    const space_taken = demo.popups.doesAgentExist(demo.mouse.grid_pos);
    if (demo.mouse.pressed() and !space_taken) {
        const p_obj = &demo.buffers.data.producers;
        const p = Producer{
            .home = demo.mouse.world_pos,
            .absolute_home = demo.mouse.grid_pos,
        };
        Producer.createAndAppend(
            gctx,
            p_obj.buf,
            &p_obj.mapping.num_structs,
            p,
        );

        const num_producers = p_obj.mapping.num_structs;
        demo.stats.setNum(gctx, num_producers, .producers);

        demo.popups.appendSquare(demo.allocator, demo.mouse.grid_pos, .producer);
        const popup = Popups.Popup{
            .grid_center = demo.mouse.grid_pos[0..2].*,
            .type_popup = Popups.PopupType.producer,
            .parameters = .{
                .producer = .{
                    .production_rate = p.production_rate,
                    .max_inventory = p.max_inventory,
                },
            },
        };
        demo.popups.appendPopup(popup);
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
    if (zgui.sliderScalar("##cs", f32, .{ .v = &demo.params.consumer_radius, .min = 1, .max = 3 })) {
        demo.buffers.vertex.circle = Shapes.createCircleVertexBuffer(
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
        const buf = demo.buffers.data.producers;
        for (0..buf.mapping.num_structs) |i| {
            const resource = gctx.lookupResource(buf.buf).?;
            const offset = i * @sizeOf(Producer) + @offsetOf(Producer, "inventory");
            gctx.queue.writeBuffer(resource, offset, i32, &.{0});
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
