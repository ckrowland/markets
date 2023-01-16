const std = @import("std");
const zgui = @import("zgui");
const main = @import("main.zig");
const Parameter = @import("parameter.zig");
const Plot = @import("plot.zig");
const Waves = @import("wave.zig");
const DemoState = main.DemoState;

pub const Args = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    margin: f32,
    no_margin: struct {
        top: bool = false,
        bottom: bool = false,
        left: bool = false,
        right: bool = false,
    } = .{},
    label: [:0]const u8,
    flags: zgui.WindowFlags = .{},
};

pub const Parameters = Args{
    .x = 0.0,
    .y = 0.0,
    .w = 0.2,
    .h = 1.0,
    .margin = 0.03,
    .label = "##params",
};

pub const Plots = Args{
    .x = 0.2,
    .y = 0.0,
    .w = 0.8,
    .h = 1.0,
    .margin = 0.03,
    .no_margin = .{
        .left = true,
    },
    .label = "##plots",
};

pub fn parameters(demo: *DemoState) void {
    setNextWindow(demo, Parameters);
    if (zgui.begin(Parameters.label, .{
        .popen = null,
        .flags = Parameters.flags,
    })) {
        defer zgui.end();
        Parameter.setup();
        Parameter.displayFPS(demo);
        Parameter.waveInput(demo);
    }
}

pub fn plots(demo: *DemoState) void {
    setNextWindow(demo, Plots);
    if (zgui.begin(Plots.label, .{
        .popen = null,
        .flags = Plots.flags,
    })) {
        defer zgui.end();

        demo.input_one.wave.clearWave();
        demo.input_two.wave.clearWave();
        demo.output.wave.clearWave();

        demo.input_one.wave.createSinWave(demo.input_one.params);
        demo.input_two.wave.createSinWave(demo.input_two.params);
        const max_x = @max(
            demo.input_one.wave.getLastPointX(),
            demo.input_two.wave.getLastPointX()
        );
        demo.output.wave.multiplyWaves(
            &demo.input_one.wave,
            &demo.input_two.wave
        );

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

fn setNextWindow(demo: *main.DemoState, args: Args) void {
    std.debug.assert(0.0 <= args.x and args.x <= 1.0);
    std.debug.assert(0.0 <= args.y and args.y <= 1.0);
    std.debug.assert(0.0 <= args.w and args.w <= 1.0);
    std.debug.assert(0.0 <= args.h and args.h <= 1.0);
    std.debug.assert(0.0 <= args.margin and args.margin <= 1.0);
    const width = @intToFloat(f32, demo.gctx.swapchain_descriptor.width);
    const height = @intToFloat(f32, demo.gctx.swapchain_descriptor.height);
    const margin_start_x = width * (args.margin + args.x);
    const margin_start_y = height * (args.margin + args.y);
    const smaller_margin_pixels = @min(margin_start_x, margin_start_y);
    var x = width * args.x + smaller_margin_pixels;
    var y = height * args.y + smaller_margin_pixels;
    var w = width * args.w - (smaller_margin_pixels * 2);
    var h = height * args.h - (smaller_margin_pixels * 2);

    if (args.no_margin.top) {
        y -= smaller_margin_pixels;
        h += smaller_margin_pixels;
    }
    if (args.no_margin.bottom) {
        h += smaller_margin_pixels;
    }
    if (args.no_margin.left) {
        x -= smaller_margin_pixels;
        w += smaller_margin_pixels;
    }
    if (args.no_margin.right) {
        w += smaller_margin_pixels;
    }

    zgui.setNextWindowPos(.{
        .x = x,
        .y = y,
    });

    zgui.setNextWindowSize(.{
        .w = w,
        .h = h,
    });
}
