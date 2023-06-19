const std = @import("std");
const zgui = @import("zgui");
const zgpu = @import("zgpu");
const Signals = @import("main.zig");
const Window = @import("../windows.zig");
const GuiPositions = @import("guiPositions.zig");

pub fn window(demo: *Signals, gctx: *zgpu.GraphicsContext) void {
    Window.setNextWindow(gctx, GuiPositions.plot);
    if (zgui.begin("Plots", Window.window_flags)) {
        defer zgui.end();

        const plotFlags = zgui.plot.Flags{
            .no_legend = true,
        };
        if (zgui.plot.beginPlot("##first", .{ .flags = plotFlags })) {
            defer zgui.plot.endPlot();

            setup(demo);
            zgui.plot.plotLine("First", f32, .{
                .xv = demo.random.xv.items,
                .yv = demo.random.yv.items,
            });
            zgui.plot.plotLine("Second", f32, .{
                .xv = demo.input.xv.items,
                .yv = demo.input.yv.items,
            });
        }

        if (zgui.plot.beginPlot("##result", .{ .flags = plotFlags })) {
            defer zgui.plot.endPlot();

            setup(demo);
            zgui.plot.plotShaded("Result", f32, .{
                .xv = demo.output.xv.items,
                .yv = demo.output.yv.items,
            });
            zgui.plot.plotLine("ResultLine", f32, .{
                .xv = demo.output.xv.items,
                .yv = demo.output.yv.items,
            });
        }
        if (zgui.plot.beginPlot("##frequencies", .{ .flags = plotFlags })) {
            defer zgui.plot.endPlot();

            // const maxX = demo.input.getLastPointX();
            // const numPoints = @intToFloat(f32, demo.input.xv.items.len);
            // const margin = 0.1;
            // const barSize = (maxX / numPoints) - margin;
            // zgui.plot.plotBars("Result", f32, .{
                // .xv = demo.output.xv.items,
                // .yv = demo.output.yv.items,
                // .bar_size = barSize
            // });
        }
    }
}


pub fn setup(demo: *Signals) void {
    const max_x = @max(demo.random.getLastPointX(), demo.input.getLastPointX());
    setupAxes(max_x);
    setupMarkers();
}

fn setupAxes(max_x: f32) void {
    zgui.plot.setupAxis(.x1, .{
        .label = "",
        .flags = .{
            .no_tick_marks = true,
            .no_tick_labels = true,
            .auto_fit = true,
        }
    });
    zgui.plot.setupAxis(.y1, .{
        .label = "",
        .flags = .{
            .no_tick_marks = true,
            .auto_fit = true,
        }
    });

    const largest_axis = @floatCast(f64, max_x);
    zgui.plot.setupAxisLimits(.x1, .{
        .min = 0,
        .max = largest_axis,
        .cond = .always,
    });
}

fn setupMarkers() void {
    zgui.plot.pushStyleVar1i(.{
        .idx = zgui.plot.StyleVar.marker,
        .v = 1,
    });
    zgui.plot.pushStyleVar1f(.{
        .idx = zgui.plot.StyleVar.marker_size,
        .v = 5.0,
    });
}
