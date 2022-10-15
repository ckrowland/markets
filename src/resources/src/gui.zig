const main = @import("resources.zig");
const DemoState = main.DemoState;
const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const Shapes = @import("shapes.zig");
const Statistics = @import("statistics.zig");

const StagingBuffer = struct {
    slice: ?[]const u32 = null,
    buffer: wgpu.Buffer = undefined,
};

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

fn getGPUStatistics(demo: *DemoState) [Statistics.array.len]u32 {
    var buf: StagingBuffer = .{
        .slice = null,
        .buffer = demo.gctx.lookupResource(demo.buffers.data.stats_mapped).?,
    };
    buf.buffer.mapAsync(
        .{ .read = true },
        0,
        @sizeOf(u32) * Statistics.array.len,
        buffersMappedCallback,
        @ptrCast(*anyopaque, &buf)
    );
    wait_loop: while (true) {
        demo.gctx.device.tick();
        if (buf.slice == null) {
            continue :wait_loop;
        }
        break;
    }
    buf.buffer.unmap();
    Statistics.clearStatsBuffer(demo.gctx, demo.buffers.data.stats);
    return buf.slice.?[0..Statistics.array.len].*;
}

fn buffersMappedCallback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
    const usb = @ptrCast(*StagingBuffer, @alignCast(@sizeOf(usize), userdata));
    std.debug.assert(usb.slice == null);
    if (status == .success) {
        usb.slice = usb.buffer.getConstMappedRange(u32, 0, Statistics.array.len).?;
    } else {
        std.debug.print("[zgpu] Failed to map buffer (code: {any})\n", .{status});
    }
}

fn updateStats(demo: *DemoState) void {
    const current_time = @floatCast(f32, demo.gctx.stats.time);
    const stats = demo.sim.stats;
    const previous_second = stats.second;
    const diff = current_time - previous_second;
    if (diff >= 1) {
        const gpu_stats = getGPUStatistics(demo);
        const vec_stats: @Vector(4, u32) = [_]u32{ gpu_stats[0], gpu_stats[1], gpu_stats[2], stats.max_stat_recorded};
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

    zgui.text("Max Producer Inventory", .{});
    _ = zgui.sliderInt("##mi", .{ .v = &demo.sim.params.max_inventory,
                              .min = 1,
                              .max = 10000 });

    zgui.dummy(.{.w = 1.0, .h = 40.0});

    zgui.text("Number of Consumers", .{});
    _ = zgui.sliderInt("##nc", .{ .v = &demo.sim.params.num_consumers,
                              .min = 1,
                              .max = 10000 });

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
