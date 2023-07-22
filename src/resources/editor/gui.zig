const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const zmath = @import("zmath");

const Camera = @import("../../camera.zig");
const Circle = @import("../../shapes/circle.zig");
const Consumer = @import("../consumer.zig");
const ConsumerHover = @import("consumer_hover.zig");
const DemoState = @import("main.zig");
const Hover = @import("hover.zig");
const Producer = @import("../producer.zig");
const Square = @import("../../shapes/square.zig");
const Statistics = @import("../statistics.zig");
const Wgpu = @import("../wgpu.zig");
const Window = @import("../../windows.zig");
const Mouse = @import("mouse.zig");
const Popups = @import("popups.zig");

pub const State = struct {
    pub const Selection = enum {
        none,
        consumers,
        producer,
    };
    selection: Selection = .producer,
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
        .mouse = demo.mouse,
        .producers = demo.buffers.data.producer,
        .stats = demo.buffers.data.stats,
        .allocator = demo.allocator,
    });


    if (demo.running) {
        Statistics.generateAndFillRandomColor(gctx, demo.buffers.data.stats.data);
        const current_time = @floatCast(f32, gctx.stats.time);
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

    _ = switch (demo.gui.selection) {
        .none => {},
        .consumers => {
            hoverUpdate(gctx, demo);
            try addingConsumer(gctx, demo);
        },
        .producer => {
            hoverUpdate(gctx, demo);
            try addingProducer(gctx, demo);
        },
    };
}

fn hoverUpdate(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(demo.buffers.data.hover).?,
        @offsetOf(Hover, "position"),
        [2]f32,
        &.{Mouse.getWorldPosition(gctx)},
    );
}

fn addingConsumer(gctx: *zgpu.GraphicsContext, demo: *DemoState) !void {
    if (demo.mouse.down() and Mouse.onGrid(gctx) and !demo.popups.anyOpen()) {
        try addConsumer(gctx, demo);
    }
    if (demo.mouse.released() and Mouse.onGrid(gctx) and !demo.popups.anyOpen()) {
        try addConsumer(gctx, demo);
        
        const gui_id = @intCast(u32, demo.popups.popups.items.len);
        try demo.popups.appendPopup(.{
            .id = .{ .gui_id = gui_id, },
            .grid_agent_center = demo.mouse.grid_pos,
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
    const gui_id = @intCast(u32, demo.popups.popups.items.len);
    const num_consumers = Wgpu.getNumStructs(gctx, Consumer, demo.buffers.data.stats);
    const world_pos = Mouse.getWorldPosition(gctx);
    const consumer = Consumer.create(.{
        .absolute_home = Camera.getGridPosition(gctx, world_pos),
        .home = world_pos,
        .grouping_id = gui_id,
    });
    var consumers = [1]Consumer{consumer};
    Wgpu.appendBuffer(gctx, Consumer, .{
        .num_old_structs = num_consumers,
        .buf = demo.buffers.data.consumer.data,
        .structs = consumers[0..],
    });

    const consumer_hover = ConsumerHover.create(.{
        .absolute_home = Camera.getGridPosition(gctx, world_pos),
        .home = world_pos,
        .grouping_id = gui_id,
    });
    var hover_circles = [1]ConsumerHover{consumer_hover};
    Wgpu.appendBuffer(gctx, ConsumerHover, .{
        .num_old_structs = num_consumers,
        .buf = demo.buffers.data.consumer_hover.data,
        .structs = hover_circles[0..],
    });
    
    Statistics.setNumConsumers(gctx, demo.buffers.data.stats.data, num_consumers + 1);
    const corners = Popups.getGridEdges(demo.mouse.grid_pos, Popups.CLOSED_SIZE);
    try demo.popups.appendSquare(demo.allocator, .{
        .id = .{ .gui_id = gui_id },
        .grid_corners = corners,
    });
}

fn addingProducer(gctx: *zgpu.GraphicsContext, demo: *DemoState) !void {
    if (demo.mouse.pressed() and Mouse.onGrid(gctx) and !demo.popups.anyOpen()) {
        const num_producers = Wgpu.getNumStructs(gctx, Producer, demo.buffers.data.stats);
        const world_pos = Mouse.getWorldPosition(gctx);
        var producer = Producer.create(.{
            .home = world_pos,
            .absolute_home = Camera.getGridPosition(gctx, world_pos),
        });
        var producers = [1]Producer{producer};
        Wgpu.appendBuffer(gctx, Producer, .{
            .num_old_structs = num_producers,
            .buf = demo.buffers.data.producer.data,
            .structs = producers[0..],
        });
        Statistics.setNumProducers(gctx, demo.buffers.data.stats.data, num_producers + 1);

        const gui_id = @intCast(u32, demo.popups.popups.items.len);
        try demo.popups.appendPopup(.{
            .id = .{ .gui_id = gui_id, },
            .grid_agent_center = demo.mouse.grid_pos,
            .type_popup = .producer,
            .parameters = .{
                .producer = .{
                    .production_rate = producer.production_rate,
                    .max_inventory = producer.max_inventory,
                },
            },
        });
        try demo.popups.appendSquare(demo.allocator, .{
            .id = .{ .gui_id = gui_id },
            .grid_corners = Popups.getGridEdges(demo.mouse.grid_pos, Popups.CLOSED_SIZE),
        });
    }
}

fn plots(demo: *DemoState) void {
    const window_size = zgui.getWindowSize();
    const margin = 40;
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
    const buttonSize = [2]f32{ 120.0, 120.0 };
    coloredButton(
        gctx,
        &demo.gui,
        State.Selection.consumers,
        demo.consumer_texture_view,
        pressedColor,
        buttonSize,
    );
    coloredButton(
        gctx,
        &demo.gui,
        State.Selection.producer,
        demo.producer_texture_view,
        pressedColor,
        buttonSize,
    );

    zgui.text("Consumer Size", .{});
    if (zgui.sliderScalar("##cs", f32, .{ .v = &demo.params.consumer_radius, .min = 1, .max = 40 })) {
        demo.buffers.vertex.circle = Circle.createVertexBuffer(
            gctx,
            40,
            demo.params.consumer_radius,
        );
    }

    if (zgui.button("Start", .{})) {
        demo.running = true;
    }
    zgui.sameLine(.{});
    if (zgui.button("Stop", .{})) {
        demo.running = false;
    }
    zgui.sameLine(.{});
    if (zgui.button("Restart", .{})) {
        demo.running = true;
        DemoState.restartSimulation(demo, gctx);
    }

    zgui.sameLine(.{});
    if (zgui.button("Supply Shock", .{})) {
        Wgpu.setAll(gctx, Producer, Wgpu.setArgs(Producer){
            .agents = demo.buffers.data.producer,
            .stats = demo.buffers.data.stats,
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
    if (guiState.selection == buttonState) {
        zgui.pushStyleColor4f(.{ .idx = .button, .c = color });
        defer zgui.popStyleColor(.{});
        if (zgui.imageButton(id, tex_id, .{ .w = size[0], .h = size[1] })) {
            guiState.selection = .none;
        }
    } else {
        if (zgui.imageButton(id, tex_id, .{ .w = size[0], .h = size[1] })) {
            guiState.selection = buttonState;
        }
    }
}
