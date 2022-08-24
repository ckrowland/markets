const main = @import("resources.zig");
const DemoState = main.DemoState;
const std = @import("std");
const zgpu = @import("zgpu");
const zgui = zgpu.zgui;
const wgpu = zgpu.wgpu;
const Shapes = @import("shapes.zig");
const StagingBuffer = main.StagingBuffer;

pub fn update(demo: *DemoState) void {
    updateStats(demo);
    zgpu.gui.newFrame(demo.gctx.swapchain_descriptor.width, demo.gctx.swapchain_descriptor.height);
    if (zgui.begin("Settings", .{})) {
        if (zgui.beginTabBar("My Tab Bar")) {
            if (zgui.beginTabItem("Parameters")) {
                parameters(demo);
                zgui.endTabItem();
            }

            if (zgui.beginTabItem("Statistics")) {
                //plots(demo);
                zgui.endTabItem();
            }
            zgui.endTabBar();
        }
        zgui.end();
    }
}

fn getTransactionsThisSecond(demo: *DemoState) f32 {
        var buf: StagingBuffer = .{
            .slice = null,
            .buffer = demo.gctx.lookupResource(demo.num_transactions_buffer_copy).?,
        };
        buf.buffer.mapAsync(.{ .read = true }, 0, @sizeOf(f32),
                                     main.buffersMappedCallback,
                                     @ptrCast(*anyopaque, &buf));
        wait_loop: while (true) {
            demo.gctx.device.tick();
            if (buf.slice == null) {
                continue :wait_loop;
            }
            break;
        }

        const num_total_transactions = buf.slice.?[0];
        const stats = demo.sim.stats;
        const transactions_this_second = num_total_transactions - stats.num_transactions_last_second;

        demo.sim.stats.num_transactions_last_second = num_total_transactions;
        demo.num_transactions.buffer.unmap();
        return transactions_this_second;
}

fn updateStats(demo: *DemoState) void {
    const current_time = @floatCast(f32, demo.gctx.stats.time);
    const current_second = @floor(current_time);
    const stats = demo.sim.stats;
    const previous_second = stats.previous_second;
    if (previous_second < current_second) {
        const num_transactions = getTransactionsThisSecond(demo);
        demo.sim.stats.transactions_array.append(num_transactions) catch unreachable;
        demo.sim.stats.max_transactions_recorded = @maximum(num_transactions, stats.max_transactions_recorded);
        demo.sim.stats.previous_second = current_second;
    }
}

//fn plots(demo: *DemoState) void {
//    const t = @floor(@floatCast(f32, demo.gctx.stats.time));
//    //const t_idx = @floatToInt(usize, @floatCast(f32, demo.gctx.stats.time));
//    const stats = demo.sim.stats;
//    const data = stats.transactions_array.items;
//    const window_size = zgui.getWindowSize();
//    const tab_bar_height = 100;
//    const margin = 50;
//    const plot_width = window_size[0] - margin;
//    const plot_height = window_size[1] - tab_bar_height - margin;
//
//    if (zgui.beginPlot("My Plot", plot_width, plot_height)) {
//        zgui.setupXAxisLimits(0, @floatCast(f64, t + 1));
//        zgui.setupYAxisLimits(0, @floatCast(f64, stats.max_transactions_recorded + (stats.max_transactions_recorded * 0.2) + 1));
//        zgui.plotLineValues("My Line Plot", data[0..]);
//        zgui.endPlot();
//    }
//}

fn parameters(demo: *DemoState) void {
    zgui.pushItemWidth(zgui.getContentRegionAvailWidth() * 0.4);
    zgui.bulletText(
        "Average :  {d:.3} ms/frame ({d:.1} fps)",
        .{ demo.gctx.stats.average_cpu_time, demo.gctx.stats.fps },
    );
    zgui.spacing();
    _ = zgui.sliderInt("Number of Producers",
                        .{ .v = &demo.sim.params.num_producers,
                           .min = 1,
                           .max = 100});
    _ = zgui.sliderInt("Production Rate",
                        .{ .v = &demo.sim.params.production_rate,
                           .min = 1,
                           .max = 100});
    _ = zgui.sliderInt("Giving Rate",
                        .{ .v = &demo.sim.params.giving_rate,
                           .min = 1,
                           .max = 100});
    _ = zgui.sliderInt("Number of Consumers",
                        .{ .v = &demo.sim.params.num_consumers,
                           .min = 1,
                           .max = 10000});
    _ = zgui.sliderInt("Consumption Rate",
                        .{ .v = &demo.sim.params.consumption_rate,
                           .min = 1,
                           .max = 100});
    _ = zgui.sliderFloat("Moving Rate",
                        .{ .v = &demo.sim.params.moving_rate,
                           .min = 1.0,
                           .max = 20});
    if (zgui.button("Start", .{})) {
        const compute_bgl = demo.gctx.createBindGroupLayout(&.{
            zgpu.bglBuffer(0, .{ .compute = true }, .storage, true, 0),
            zgpu.bglBuffer(1, .{ .compute = true }, .read_only_storage, true, 0),
            zgpu.bglBuffer(2, .{ .compute = true }, .storage, true, 0),
        });
        defer demo.gctx.releaseResource(compute_bgl);
        demo.sim.createAgents();
        demo.producer_buffer = Shapes.createProducerBuffer(demo.gctx, demo.sim.producers);
        demo.consumer_buffer = Shapes.createConsumerBuffer(demo.gctx, demo.sim.consumers);
        demo.consumer_bind_group = Shapes.createBindGroup(demo.gctx,
                                                    demo.sim,
                                                    compute_bgl,
                                                    demo.consumer_buffer,
                                                    demo.producer_buffer,
                                                    demo.num_transactions_buffer);
    }
}
