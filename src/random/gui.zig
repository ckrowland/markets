const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Statistics = @import("statistics.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Wgpu = @import("wgpu.zig");
const Window = @import("windows.zig");
const Circle = @import("circle.zig");
const Callbacks = @import("callbacks.zig");

pub fn update(demo: *DemoState) void {
    const gctx = demo.gctx;
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

    Wgpu.runCallbackIfReady(u32, &demo.buffers.data.stats.mapping);
    Wgpu.runCallbackIfReady(Producer, &demo.buffers.data.producers.mapping);
    Wgpu.runCallbackIfReady(Consumer, &demo.buffers.data.consumers.mapping);

    if (demo.running) {
        gctx.queue.writeBuffer(
            gctx.lookupResource(demo.buffers.data.stats.buf).?,
            3 * @sizeOf(u32),
            f32,
            &.{ random.float(f32), random.float(f32), random.float(f32) },
        );
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

fn plots(demo: *DemoState) void {
    const window_size = zgui.getWindowSize();
    const margin = 15;
    const plot_width = window_size[0] - margin;
    const plot_height = window_size[1] - margin;

    if (zgui.plot.beginPlot("", .{
        .w = plot_width,
        .h = plot_height,
        .flags = .{},
    })) {
        zgui.plot.setupAxis(.x1, .{
            .label = "",
            .flags = .{ .auto_fit = true },
        });
        zgui.plot.setupAxis(.y1, .{
            .label = "",
            .flags = .{ .auto_fit = true },
        });
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
    zgui.bulletText(
        "{d:.3} ms/frame ({d:.1} fps)",
        .{ demo.gctx.stats.average_cpu_time, demo.gctx.stats.fps },
    );

    zgui.text("Number Of Producers", .{});

    const p_bufs = demo.buffers.data.producers;
    if (zgui.sliderScalar(
        "##np",
        u32,
        .{ .v = &demo.params.num_producers.new, .min = 1, .max = 100 },
    )) {
        const num_producers = demo.params.num_producers;
        Statistics.setNum(gctx, .{
            .stat_obj = demo.buffers.data.stats,
            .num = num_producers.new,
            .param = .producers,
        });

        if (num_producers.old >= num_producers.new) {
            Wgpu.shrinkBuffer(gctx, Producer, .{
                .new_size = num_producers.new,
                .buf = p_bufs.buf,
            });
            demo.buffers.data.producers.list.resize(num_producers.new) catch unreachable;
            demo.buffers.data.producers.mapping.num_structs = num_producers.new;
        } else {
            Producer.generateBulk(
                gctx,
                &demo.buffers.data.producers,
                demo.params,
                num_producers.new - num_producers.old,
            );
        }
        demo.params.num_producers.old = demo.params.num_producers.new;
    }

    zgui.text("Production Rate", .{});
    if (zgui.sliderScalar(
        "##pr",
        u32,
        .{ .v = &demo.params.production_rate, .min = 1, .max = 1000 },
    )) {
        for (demo.buffers.data.producers.list.items, 0..) |_, i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.producers.buf).?,
                i * @sizeOf(Producer) + @offsetOf(Producer, "production_rate"),
                u32,
                &.{demo.params.production_rate},
            );
        }
    }

    zgui.text("Demand Rate", .{});
    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted(
            "How much consumers take from producers on a trip.",
        );
        zgui.endTooltip();
    }

    if (zgui.sliderScalar(
        "##dr",
        u32,
        .{ .v = &demo.params.demand_rate, .min = 1, .max = 1000 },
    )) {
        for (demo.buffers.data.consumers.list.items, 0..) |_, i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.consumers.buf).?,
                i * @sizeOf(Consumer) + @offsetOf(Consumer, "demand_rate"),
                u32,
                &.{demo.params.demand_rate},
            );
        }
    }

    zgui.text("Max Producer Inventory", .{});
    if (zgui.sliderScalar("##mi", u32, .{
        .v = &demo.params.max_inventory,
        .min = 10,
        .max = 10000,
    })) {
        for (demo.buffers.data.producers.list.items, 0..) |_, i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.producers.buf).?,
                i * @sizeOf(Producer) + @offsetOf(Producer, "max_inventory"),
                u32,
                &.{demo.params.max_inventory},
            );
        }
    }

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    const c_bufs = demo.buffers.data.consumers;
    zgui.text("Number of Consumers", .{});
    if (zgui.sliderScalar("##nc", u32, .{
        .v = &demo.params.num_consumers.new,
        .min = 1,
        .max = 10000,
    })) {
        const num_consumers = demo.params.num_consumers;
        Statistics.setNum(gctx, .{
            .stat_obj = demo.buffers.data.stats,
            .num = num_consumers.new,
            .param = .consumers,
        });

        if (num_consumers.old >= num_consumers.new) {
            Wgpu.shrinkBuffer(gctx, Consumer, .{
                .new_size = num_consumers.new,
                .buf = c_bufs.buf,
            });
            demo.buffers.data.consumers.list.resize(num_consumers.new) catch unreachable;
            demo.buffers.data.consumers.mapping.num_structs = num_consumers.new;
        } else {
            Consumer.generateBulk(
                gctx,
                &demo.buffers.data.consumers,
                demo.params,
                num_consumers.new - num_consumers.old,
            );
        }
        demo.params.num_consumers.old = demo.params.num_consumers.new;
    }

    zgui.text("Moving Rate", .{});
    if (zgui.sliderScalar("##mr", f32, .{
        .v = &demo.params.moving_rate,
        .min = 1.0,
        .max = 20,
    })) {
        for (demo.buffers.data.consumers.list.items, 0..) |_, i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.consumers.buf).?,
                i * @sizeOf(Consumer) + @offsetOf(Consumer, "moving_rate"),
                f32,
                &.{demo.params.moving_rate},
            );
        }
    }

    zgui.text("Consumer Size", .{});
    if (zgui.sliderScalar("##cs", f32, .{
        .v = &demo.params.consumer_radius,
        .min = 1,
        .max = 40,
    })) {
        demo.buffers.vertex.circle = Circle.createVertexBuffer(
            gctx,
            40,
            demo.params.consumer_radius,
        );
    }

    if (zgui.button("Start", .{})) {
        demo.running = true;
    }
    if (zgui.button("Stop", .{})) {
        demo.running = false;
    }
    if (zgui.button("Restart", .{})) {
        demo.running = true;
        Main.restartSimulation(demo);
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
