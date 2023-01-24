const DemoState = @import("main.zig");
const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const Statistics = @import("statistics.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Circle = @import("circle.zig");
const Square = @import("square.zig");
const Wgpu = @import("wgpu.zig");
const Window = @import("../windows.zig");

pub fn update(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    if (demo.running) {
        updateStats(demo, gctx);
    }

    //zgui.showDemoWindow(null);

    Window.setNextWindow(gctx, Window.Args{
        .x = 0.0,
        .y = 0.13,
        .w = 0.25,
        .h = 0.62,
        .margin = 0.02,
    });
    if (zgui.begin("Parameters", Window.window_flags)) {
        zgui.pushIntId(2);
        parameters(demo, gctx);
        zgui.popId();
    }
    zgui.end();

    Window.setNextWindow(gctx, Window.Args{
        .x = 0.0,
        .y = 0.75,
        .w = 1.0,
        .h = 0.25,
        .margin = 0.02,
        .no_margin = .{ .top = true },
    });
    if (zgui.begin("Data", Window.window_flags)) {
        zgui.pushIntId(3);
        plots(demo);
        zgui.popId();
    }
    zgui.end();
}

fn updateStats(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    const current_time = @floatCast(f32, gctx.stats.time);
    const previous_second = demo.stats.second;
    const diff = current_time - previous_second;
    if (diff >= 1) {
        const gpu_stats = Statistics.getGPUStatistics(demo, gctx);
        demo.stats.second = current_time;
        demo.stats.num_transactions.append(gpu_stats[0]) catch unreachable;

        const consumers = Consumer.getAll(demo, gctx);
        var empty_consumers: u32 = 0;
        for (consumers) |c| {
            if (c.inventory == 0) {
                empty_consumers += 1;
            }
        }
        demo.stats.num_empty_consumers.append(empty_consumers) catch unreachable;

        const producers = Producer.getAll(demo, gctx);
        var total_inventory: u32 = 0;
        for (producers) |p| {
            total_inventory += p.inventory;
        }
        demo.stats.num_total_producer_inventory.append(total_inventory) catch unreachable;
    }
}

fn plots(demo: *DemoState) void {
    const nt = demo.stats.num_transactions.items;
    const nec = demo.stats.num_empty_consumers.items;
    const tpi = demo.stats.num_total_producer_inventory.items;
    const window_size = zgui.getWindowSize();
    const margin = 40;
    const plot_width = window_size[0] - margin;
    const plot_height = window_size[1] - margin;
    const plot_flags = .{ .w = plot_width, .h = plot_height, .flags = .{} };

    if (zgui.plot.beginPlot("", plot_flags)) {
        zgui.plot.setupAxis(.x1, .{ .label = "", .flags = .{ .auto_fit = true } });
        zgui.plot.setupAxis(.y1, .{ .label = "", .flags = .{ .auto_fit = true } });
        zgui.plot.setupLegend(.{ .north = true, .west = true }, .{});
        zgui.plot.plotLineValues("Transactions", u32, .{ .v = nt[0..] });
        zgui.plot.plotLineValues("Empty Consumers", u32, .{ .v = nec[0..] });
        zgui.plot.plotLineValues("Total Producer Inventory", u32, .{ .v = tpi[0..] });
        zgui.plot.endPlot();
    }
}

fn parameters(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
    zgui.text("Number Of Producers", .{});
    if (zgui.sliderScalar(
        "##np",
        u32,
        .{ .v = &demo.params.num_producers, .min = 1, .max = 100 },
    )) {
        const old_producers = Producer.getAll(demo, gctx);
        if (demo.params.num_producers > old_producers.len) {
            Producer.add(demo, gctx);
        } else {
            Producer.remove(demo, gctx);
        }
    }

    zgui.text("Production Rate", .{});
    if (zgui.sliderScalar(
        "##pr",
        u32,
        .{ .v = &demo.params.production_rate, .min = 1, .max = 1000 },
    )) {
        Producer.setAll(demo, gctx, Producer.Parameter.production_rate);
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
        Consumer.setAll(demo, gctx, Consumer.Parameter.demand_rate);
    }

    zgui.text("Max Producer Inventory", .{});
    if (zgui.sliderScalar("##mi", u32, .{ .v = &demo.params.max_inventory, .min = 10, .max = 10000 })) {
        Producer.setAll(demo, gctx, Producer.Parameter.max_inventory);
    }

    zgui.text("Producer Size", .{});
    if (zgui.sliderScalar("##pw", f32, .{ .v = &demo.params.producer_width, .min = 10, .max = 70 })) {
        demo.buffers.vertex.square = Square.createVertexBuffer(gctx, demo.params.producer_width);
    }
    zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    zgui.text("Number of Consumers", .{});
    if (zgui.sliderScalar("##nc", u32, .{ .v = &demo.params.num_consumers, .min = 1, .max = 10000 })) {
        const old_consumers = Consumer.getAll(demo, gctx);

        if (demo.params.num_consumers > old_consumers.len) {
            Consumer.add(demo, gctx);
        } else {
            Consumer.remove(demo, gctx);
        }
    }

    zgui.text("Moving Rate", .{});
    if (zgui.sliderScalar("##mr", f32, .{ .v = &demo.params.moving_rate, .min = 1.0, .max = 20 })) {
        Consumer.setAll(demo, gctx, Consumer.Parameter.moving_rate);
    }

    zgui.text("Consumer Size", .{});
    if (zgui.sliderScalar("##cs", f32, .{ .v = &demo.params.consumer_radius, .min = 1, .max = 40 })) {
        demo.buffers.vertex.circle = Circle.createVertexBuffer(gctx, demo.params.consumer_radius);
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
        Producer.setAll(demo, gctx, Producer.Parameter.supply_shock);
    }

    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("Set all producer inventory to 0.");
        zgui.endTooltip();
    }
}
