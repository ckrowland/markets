const main = @import("resources.zig");
const DemoState = main.DemoState;
const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const Statistics = @import("statistics.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Circle = @import("circle.zig");
const Wgpu = @import("wgpu.zig");

pub fn update(demo: *DemoState) void {
    updateStats(demo);

    const width = demo.gctx.swapchain_descriptor.width;
    const height = demo.gctx.swapchain_descriptor.height;

    zgui.backend.newFrame(width, height);

    const window_width = @intToFloat(f32, width);
    const window_height = @intToFloat(f32, height);
    const margin: f32 = 40;
    const params_width: f32 = window_width / 5;
    const stats_height: f32 = window_height / 5 + 30;
    zgui.setNextWindowPos(.{ .x = margin,
                            .y = margin,
                            .cond = zgui.Condition.always });
    zgui.setNextWindowSize(.{ .w = params_width,
                            .h = window_height - stats_height - (margin * 3),
                            .cond = zgui.Condition.always });
    const zgui_window_flags = .{
        .flags = zgui.WindowFlags.no_decoration,
    };

    if (zgui.begin("Parameters", zgui_window_flags)) {
        zgui.pushIntId(1);
        parameters(demo);
        zgui.popId();
    }
    zgui.end();

    zgui.setNextWindowPos(.{ .x = margin,
                            .y = window_height - stats_height - margin,
                            .cond = zgui.Condition.always });
    zgui.setNextWindowSize(.{ .w = window_width - (2 * margin),
                            .h = stats_height,
                            .cond = zgui.Condition.always });

    if (zgui.begin("Data", zgui_window_flags)) {
        zgui.pushIntId(2);
        plots(demo);
        zgui.popId();
    }
    zgui.end();
}

fn updateStats(demo: *DemoState) void {
    const current_time = @floatCast(f32, demo.gctx.stats.time);
    const previous_second = demo.stats.second;
    const diff = current_time - previous_second;
    if (diff >= 1) {
        const gpu_stats = Statistics.getGPUStatistics(demo);
        demo.stats.second = current_time;
        demo.stats.num_transactions.append(gpu_stats[0]) catch unreachable;

        
        const consumers = Consumer.getAll(demo);
        var empty_consumers: u32 = 0;
        for (consumers) |c| {
           if (c.inventory == 0) {
                empty_consumers += 1; 
           }
        }
        demo.stats.num_empty_consumers.append(empty_consumers) catch unreachable;

        const producers = Producer.getAll(demo);
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

    if (zgui.plot.beginPlot("", plot_flags)){
        zgui.plot.setupAxis(.x1, .{ .label = "", .flags = .{ .auto_fit = true }});
        zgui.plot.setupAxis(.y1, .{ .label = "", .flags = .{ .auto_fit = true }});
        zgui.plot.setupLegend(.{ .north = true, .west = true }, .{});
        zgui.plot.plotLineValues("Transactions", u32, .{ .v = nt[0..] });
        zgui.plot.plotLineValues("Empty Consumers", u32, .{ .v = nec[0..] });
        zgui.plot.plotLineValues("Total Producer Inventory", u32, .{ .v = tpi[0..]});
        zgui.plot.endPlot();
    }
}

fn parameters(demo: *DemoState) void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
    zgui.bulletText("{d:.1} fps", .{ demo.gctx.stats.fps });
    zgui.spacing();
    zgui.text("Number Of Producers", .{});
    if(zgui.sliderScalar(
        "##np",
        u32,
        .{ .v = &demo.params.num_producers, .min = 1, .max = 100 },
    )) {
        const old_producers = Producer.getAll(demo);
        if (demo.params.num_producers > old_producers.len){
            Producer.add(demo);
        } else {
            Producer.remove(demo);
        }
    }

    zgui.text("Production Rate", .{});
    if(zgui.sliderScalar(
        "##pr",
        u32,
        .{ .v = &demo.params.production_rate, .min = 1, .max = 1000 },
    )) {
        Producer.setAll(demo, Producer.Parameter.production_rate);
    }

    zgui.text("Giving Rate", .{});
    if(zgui.sliderScalar(
        "##gr",
        u32,
        .{ .v = &demo.params.giving_rate, .min = 1, .max = 1000 },
    )) {
        Producer.setAll(demo, Producer.Parameter.giving_rate);
    }

    zgui.text("Max Producer Inventory", .{});
    if(zgui.sliderScalar(
        "##mi",
        u32,
        .{ .v = &demo.params.max_inventory, .min = 10, .max = 10000 }
    )) {
        Producer.setAll(demo, Producer.Parameter.max_inventory);
    }

    zgui.dummy(.{.w = 1.0, .h = 40.0});

    zgui.text("Number of Consumers", .{});
    if(zgui.sliderScalar(
        "##nc", 
        u32,
        .{ .v = &demo.params.num_consumers, .min = 1, .max = 10000 }
    )) {
        const old_consumers = Consumer.getAll(demo);

        if (demo.params.num_consumers > old_consumers.len){
            Consumer.add(demo);
        } else {
            Consumer.remove(demo);
        }
    }

    zgui.text("Moving Rate", .{});
    if(zgui.sliderScalar(
        "##mr",
        f32,
        .{ .v = &demo.params.moving_rate, .min = 1.0, .max = 20 }
    )) {
        Consumer.setAll(demo, Consumer.Parameter.moving_rate);
    }

    zgui.text("Consumer Size", .{});
    if(zgui.sliderScalar("##cs", f32, .{
        .v = &demo.params.consumer_radius,
        .min = 1,
        .max = 20
    })) {
        demo.buffers.vertex.circle = Circle.createVertexBuffer(demo.gctx, demo.params.consumer_radius);
    }

    if (zgui.button("Restart", .{})) {
        main.restartSimulation(demo);
    }

    zgui.dummy(.{.w = 1.0, .h = 40.0});

    if (zgui.button("Supply Shock", .{})) {
        Producer.setAll(demo, Producer.Parameter.supply_shock);
    }

    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("Set all producer inventory to 0");
        zgui.endTooltip();
    }
}
