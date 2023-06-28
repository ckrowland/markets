const DemoState = @import("main.zig");
const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const Statistics = @import("statistics.zig");
const Consumer = @import("../consumer.zig");
const Producer = @import("../producer.zig");
const Wgpu = @import("../wgpu.zig");
const Window = @import("../../windows.zig");
const Square = @import("../../shapes/square.zig");
const Circle = @import("../../shapes/circle.zig");

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
            demo.stats.update(gctx, demo);
        }
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
    zgui.text("Number Of Producers", .{});

    const p_bufs = demo.buffers.data.producer;
    if (zgui.sliderScalar(
        "##np",
        u32,
        .{ .v = &demo.params.num_producers.new, .min = 1, .max = 100 },
    )) {
        const num_producers = demo.params.num_producers;
        Statistics.setNumProducers(gctx, demo.buffers.data.stats.data, num_producers.new);

        if (num_producers.old >= num_producers.new) {
            Wgpu.shrinkBuffer(gctx, Producer, .{
                .new_size = num_producers.new,
                .buf = p_bufs.data,
            });
        } else {
            const num_new = num_producers.new - num_producers.old;
            var producers: [DemoState.MAX_NUM_PRODUCERS]Producer = undefined;
            const p_len = Producer.createBulk(&producers, demo.params, num_new);
            Wgpu.appendBuffer(gctx, Producer, .{
                .num_old_structs = num_producers.old,
                .buf = demo.buffers.data.producer.data,
                .structs = producers[0..p_len],
            });
        }
        demo.params.num_producers.old = demo.params.num_producers.new;
    }

    zgui.text("Production Rate", .{});
    if (zgui.sliderScalar(
        "##pr",
        u32,
        .{ .v = &demo.params.production_rate, .min = 1, .max = 1000 },
    )) {
        Wgpu.setAll(gctx, Producer, .{
            .agents = demo.buffers.data.producer,
            .stats = demo.buffers.data.stats,
            .parameter = .{
                .production_rate = demo.params.production_rate,
            },
        });
    }

    zgui.text("Demand Rate", .{});
    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("How much consumers demand from producers on a single trip.");
        zgui.endTooltip();
    }

    if (zgui.sliderScalar(
        "##dr",
        u32,
        .{ .v = &demo.params.demand_rate, .min = 1, .max = 1000 },
    )) {
        Wgpu.setAll(gctx, Consumer, .{
            .agents = demo.buffers.data.consumer,
            .stats = demo.buffers.data.stats,
            .parameter = .{
                .demand_rate = demo.params.demand_rate,
            },
        });
    }

    zgui.text("Max Producer Inventory", .{});
    if (zgui.sliderScalar("##mi", u32, .{ .v = &demo.params.max_inventory, .min = 10, .max = 10000 })) {
        Wgpu.setAll(gctx, Producer, .{
            .agents = demo.buffers.data.producer,
            .stats = demo.buffers.data.stats,
            .parameter = .{
                .max_inventory = demo.params.max_inventory,
            },
        });
    }

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    const c_bufs = demo.buffers.data.consumer;
    zgui.text("Number of Consumers", .{});
    if (zgui.sliderScalar("##nc", u32, .{ .v = &demo.params.num_consumers.new, .min = 1, .max = 10000 })) {
        const num_consumers = demo.params.num_consumers;
        Statistics.setNumConsumers(gctx, demo.buffers.data.stats.data, num_consumers.new);

        if (num_consumers.old >= num_consumers.new) {
            Wgpu.shrinkBuffer(gctx, Consumer, .{
                .new_size = num_consumers.new,
                .buf = c_bufs.data,
            });
        } else {
            const num_new = num_consumers.new - num_consumers.old;
            var consumers: [DemoState.MAX_NUM_CONSUMERS]Consumer = undefined;
            const c_len = Consumer.createRandomBulk(&consumers, demo.params, num_new);
            Wgpu.appendBuffer(gctx, Consumer, .{
                .num_old_structs = num_consumers.old,
                .buf = demo.buffers.data.consumer.data,
                .structs = consumers[0..c_len],
            });
        }
        demo.params.num_consumers.old = demo.params.num_consumers.new;
    }

    zgui.text("Moving Rate", .{});
    if (zgui.sliderScalar("##mr", f32, .{ .v = &demo.params.moving_rate, .min = 1.0, .max = 20 })) {
        Wgpu.setAll(gctx, Consumer, .{
            .agents = demo.buffers.data.consumer,
            .stats = demo.buffers.data.stats,
            .parameter = .{
                .moving_rate = demo.params.moving_rate,
            },
        });
    }

    zgui.text("Consumer Size", .{});
    if (zgui.sliderScalar("##cs", f32, .{ .v = &demo.params.consumer_radius, .min = 1, .max = 40 })) {
        demo.buffers.vertex.circle = Circle.createVertexBuffer(gctx, 40, demo.params.consumer_radius);
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
        Wgpu.setAll(gctx, Producer, .{
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
