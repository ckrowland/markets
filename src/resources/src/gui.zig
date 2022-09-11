const main = @import("resources.zig");
const GPUStats = main.GPUStats;
const DemoState = main.DemoState;
const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const Shapes = @import("shapes.zig");
const StagingBuffer = main.StagingBuffer;
const Statistics = @import("simulation.zig").Statistics;

pub fn update(demo: *DemoState) void {
    updateStats(demo);

    zgui.backend.newFrame(demo.gctx.swapchain_descriptor.width, demo.gctx.swapchain_descriptor.height);

    const window_width = @intToFloat(f32, demo.gctx.swapchain_descriptor.width);
    const window_height = @intToFloat(f32, demo.gctx.swapchain_descriptor.height);
    const margin: f32 = 40;
    const stats_height: f32 = 400;
    const params_width: f32 = 600;
    zgui.setNextWindowPos(.{ .x = margin,
                            .y = margin,
                            .cond = zgui.Condition.once });
    zgui.setNextWindowSize(.{ .w = params_width,
                            .h = window_height - stats_height - (margin * 3),
                            .cond = zgui.Condition.once });

    if (zgui.begin("Parameters", .{})) {
        zgui.pushIntId(1);
        parameters(demo);
        zgui.popId();
    }
    zgui.end();

    zgui.setNextWindowPos(.{ .x = margin,
                            .y = window_height - stats_height - margin,
                            .cond = zgui.Condition.once });
    zgui.setNextWindowSize(.{ .w = window_width - (2 * margin),
                            .h = stats_height,
                            .cond = zgui.Condition.once });

    if (zgui.begin("Data", .{})) {
        zgui.pushIntId(2);
        plots(demo);
        zgui.popId();
    }
    zgui.end();
}

fn getGPUStatistics(demo: *DemoState) [3]i32 {
    var buf: StagingBuffer = .{
        .slice = null,
        .buffer = demo.gctx.lookupResource(demo.stats_mapped_buffer).?,
    };
    buf.buffer.mapAsync(.{ .read = true },
                        0,
                        @sizeOf(i32) * 3,
                        main.buffersMappedCallback,
                        @ptrCast(*anyopaque, &buf));
    wait_loop: while (true) {
        demo.gctx.device.tick();
        if (buf.slice == null) {
            continue :wait_loop;
        }
        break;
    }

    const stats_data = [_][3]i32{ [3]i32{ 0, 0, 0 }, };
    demo.gctx.queue.writeBuffer(demo.gctx.lookupResource(demo.stats_buffer).?, 0, [3]i32, stats_data[0..]);
    demo.stats.buffer.unmap();
    return buf.slice.?[0];
}

fn updateStats(demo: *DemoState) void {
    const current_time = @floatCast(f32, demo.gctx.stats.time);
    const stats = demo.sim.stats;
    const previous_second = stats.second;
    const diff = current_time - previous_second;
    if (diff > 0.5) {
        const gpu_stats = getGPUStatistics(demo);
        const vec_stats: @Vector(4, i32) = [_]i32{ gpu_stats[0], gpu_stats[1], gpu_stats[2], stats.max_stat_recorded};
        const max_stat = @reduce(.Max, vec_stats);
        demo.sim.stats.num_transactions.append(gpu_stats[0]) catch unreachable;
        demo.sim.stats.second = current_time;
        demo.sim.stats.max_stat_recorded = max_stat;
        demo.sim.stats.num_empty_consumers.append(gpu_stats[1]) catch unreachable;
        demo.sim.stats.num_total_producer_inventory.append(gpu_stats[2]) catch unreachable;
    }
}

fn plots(demo: *DemoState) void {
    const stats = demo.sim.stats;
    const nt = stats.num_transactions.items;
    const nec = stats.num_empty_consumers.items;
    const tpi = stats.num_total_producer_inventory.items;
    const window_size = zgui.getWindowSize();
    const tab_bar_height = 50;
    const margin = 40;
    const plot_width = window_size[0] - margin;
    const plot_height = window_size[1] - tab_bar_height - margin;

    if (zgui.beginPlot("", .{ .w = plot_width, .h = plot_height, .flags = .{}})) {
        zgui.setupXAxis("", .{ .auto_fit = true, });
        zgui.setupYAxis("", .{ .auto_fit = true });
        zgui.setupLegend(zgui.PlotLocation.north_west, .{});
        zgui.plotLineValues("Transactions", .{ .slice = nt[0..], .flags = .{}});
        zgui.plotLineValues("Empty Consumers", .{ .slice = nec[0..],
                                                  .flags = .{.no_clip = true} });
        zgui.plotLineValues("Total Producer Inventory", .{ .slice = tpi[0..],
                                                           .flags = .{}});
        zgui.endPlot();
    }
}

fn parameters(demo: *DemoState) void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
    zgui.bulletText("{d:.1} fps", .{ demo.gctx.stats.fps });
    zgui.spacing();
    zgui.text("Number Of Producers", .{});
    _ = zgui.sliderInt("##np", .{ .v = &demo.sim.params.num_producers,
                              .min = 1,
                              .max = 100 });

    zgui.text("Production Rate", .{});
    _ = zgui.sliderInt("##pr", .{ .v = &demo.sim.params.production_rate,
                              .min = 1,
                              .max = 1000 });

    zgui.text("Giving Rate", .{});
    _ = zgui.sliderInt("##gr", .{ .v = &demo.sim.params.giving_rate,
                              .min = 1,
                              .max = 1000 });

    zgui.text("Max Inventory", .{});
    _ = zgui.sliderInt("##mi", .{ .v = &demo.sim.params.max_inventory,
                              .min = 1,
                              .max = 10000 });

    zgui.dummy(.{.w = 1.0, .h = 40.0});

    zgui.text("Number of Consumers", .{});
    _ = zgui.sliderInt("##nc", .{ .v = &demo.sim.params.num_consumers,
                              .min = 1,
                              .max = 10000 });

    zgui.text("Consumption Rate", .{});
    _ = zgui.sliderInt("##cr", .{ .v = &demo.sim.params.consumption_rate,
                              .min = 1,
                              .max = 100 });

    zgui.text("Moving Rate", .{});
    _ = zgui.sliderFloat("##mr", .{ .v = &demo.sim.params.moving_rate,
                                .min = 1.0,
                                .max = 20 });

    zgui.text("Consumer Size", .{});
    _ = zgui.sliderFloat("##cs", .{ .v = &demo.sim.params.consumer_radius,
                                .min = 1,
                                .max = 20 });

    if (zgui.button("Start", .{})) {
        main.startSimulation(demo);
    }

    zgui.dummy(.{.w = 1.0, .h = 40.0});

    if (zgui.button("Supply Shock", .{})) {
        main.supplyShock(demo);
    }
}
