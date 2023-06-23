const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;

const Camera = @import("../../camera.zig");
const Circle = @import("../../shapes/circle.zig");
const Consumer = @import("../consumer.zig");
const DemoState = @import("main.zig");
const Hover = @import("hover.zig");
const Producer = @import("../producer.zig");
const Square = @import("../../shapes/square.zig");
const Statistics = @import("../statistics.zig");
const Wgpu = @import("../wgpu.zig");
const Window = @import("../../windows.zig");
const Mouse = @import("input.zig");
const Popups = @import("popups.zig");

pub const Selection = enum {
    none,
    consumers,
    producer,
};

pub fn update(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
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

    // Popups.display();

    if (demo.running) {
        gctx.queue.writeBuffer(
            gctx.lookupResource(demo.buffers.data.stats.data).?,
            3 * @sizeOf(u32),
            f32,
            &.{ random.float(f32), random.float(f32), random.float(f32) },
        );
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

    _ = switch (demo.placing) {
        .none => {},
        .consumers => {
            hoverUpdate(gctx, demo);
            addingConsumer(gctx, demo);
        },
        .producer => {
            hoverUpdate(gctx, demo);
            addingProducer(gctx, demo);
        },
    };
}

fn hoverUpdate(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(demo.buffers.data.hover).?,
        @offsetOf(Hover, "position"),
        [4]f32,
        &.{Mouse.getWorldPosition(gctx)},
    );
}

fn addingConsumer(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    if (demo.mouse.down() and Mouse.onGrid(gctx)) {
        const num_consumers = Wgpu.getNumStructs(gctx, Consumer, demo.buffers.data.stats);
        const world_pos = Mouse.getWorldPosition(gctx);
        // We can probably use Wgpu.appendBuffer()
        gctx.queue.writeBuffer(
            gctx.lookupResource(demo.buffers.data.consumer.data).?,
            @sizeOf(Consumer) * num_consumers,
            Consumer,
            &.{Consumer.create(.{
                .home = world_pos,
                .absolute_home = Camera.getGridPosition(gctx, world_pos),
            })},
        );
        Statistics.setNumConsumers(gctx, demo.buffers.data.stats.data, num_consumers + 1);
    }
}

fn addingProducer(gctx: *zgpu.GraphicsContext, demo: *DemoState) void {
    if (demo.mouse.pressed() and Mouse.onGrid(gctx)) {
        const num_producers = Wgpu.getNumStructs(gctx, Producer, demo.buffers.data.stats);
        const world_pos = Mouse.getWorldPosition(gctx);
        gctx.queue.writeBuffer(
            gctx.lookupResource(demo.buffers.data.producer.data).?,
            @sizeOf(Producer) * num_producers,
            Producer,
            &.{Producer.create(.{
                .position = world_pos,
                .absolute_position = Camera.getGridPosition(gctx, world_pos),
            })},
        );
        Statistics.setNumProducers(gctx, demo.buffers.data.stats.data, num_producers + 1);

        // Create zgui window at mouse position
        const cursor_pos = gctx.window.getCursorPos();
        demo.popups.addWindow(.{
            .x = @floatCast(f32, cursor_pos[0]),
            .y = @floatCast(f32, cursor_pos[1]),
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
        &demo.placing,
        Selection.consumers,
        demo.consumerTextureView,
        pressedColor,
        buttonSize,
    );
    coloredButton(
        gctx,
        &demo.placing,
        Selection.producer,
        demo.producerTextureView,
        pressedColor,
        buttonSize,
    );

    zgui.text("Number Of Producers", .{});

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
            .get_buffer = demo.buffers.data.producer,
            .stats = demo.buffers.data.stats,
            .num_agents = demo.params.num_producers.new,
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
    guiState: *Selection,
    buttonState: Selection,
    textureView: zgpu.TextureViewHandle,
    color: [4]f32,
    size: [2]f32,
) void {
    const tex_id = gctx.lookupResource(textureView).?;
    const id = @tagName(buttonState);
    if (guiState.* == buttonState) {
        zgui.pushStyleColor4f(.{ .idx = .button, .c = color });
        defer zgui.popStyleColor(.{});
        if (zgui.imageButton(id, tex_id, .{ .w = size[0], .h = size[1] })) {
            guiState.* = Selection.none;
        }
    } else {
        if (zgui.imageButton(id, tex_id, .{ .w = size[0], .h = size[1] })) {
            guiState.* = buttonState;
        }
    }
}
