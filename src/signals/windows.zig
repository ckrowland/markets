const std = @import("std");
const zgui = @import("zgui");
const zgpu = @import("zgpu");
const Signals = @import("main.zig");
const Parameter = @import("parameter.zig");
const Plot = @import("plot.zig");
const Waves = @import("wave.zig");
const Window = @import("../windows.zig");

pub fn parameters(demo: *Signals, gctx: *zgpu.GraphicsContext) void {
    Window.setNextWindow(gctx, Window.Args{
        .x = 0.0,
        .y = 0.13,
        .w = 0.25,
        .h = 0.62,
        .margin = 0.02,
    });
    if (zgui.begin("Parameters", Window.window_flags)) {
        defer zgui.end();
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        Parameter.waveInput(demo);
    }
}

pub fn plots(demo: *Signals, gctx: *zgpu.GraphicsContext) void {
    Window.setNextWindow(gctx, Window.Args{
        .x = 0.25,
        .y = 0.0,
        .w = 0.75,
        .h = 1.0,
        .margin = 0.02,
        .no_margin = .{
            .left = true,
        },
    });
    if (zgui.begin("Plots", Window.window_flags)) {
        defer zgui.end();

        demo.input_one.wave.clearWave();
        demo.input_two.wave.clearWave();
        demo.output.wave.clearWave();

        demo.input_one.wave.createSinWave(demo.input_one.params);
        demo.input_two.wave.createSinWave(demo.input_two.params);
        const max_x = @max(demo.input_one.wave.getLastPointX(), demo.input_two.wave.getLastPointX());
        demo.output.wave.multiplyWaves(&demo.input_one.wave, &demo.input_two.wave);

        if (zgui.plot.beginPlot("##first", .{})) {
            defer zgui.plot.endPlot();

            Plot.setup(max_x);
            zgui.plot.plotLine("Sin", f32, .{
                .xv = demo.input_one.wave.xv.items,
                .yv = demo.input_one.wave.yv.items,
            });
        }

        if (zgui.plot.beginPlot("##second", .{})) {
            defer zgui.plot.endPlot();

            Plot.setup(max_x);
            zgui.plot.plotLine("Sin", f32, .{
                .xv = demo.input_two.wave.xv.items,
                .yv = demo.input_two.wave.yv.items,
            });
        }

        if (zgui.plot.beginPlot("##third", .{})) {
            defer zgui.plot.endPlot();

            Plot.setup(max_x);
            zgui.plot.plotLine("Sin", f32, .{
                .xv = demo.output.wave.xv.items,
                .yv = demo.output.wave.yv.items,
            });
        }
    }
}
