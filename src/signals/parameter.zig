const std = @import("std");
const zgui = @import("zgui");
const zgpu = @import("zgpu");
const Signals = @import("main.zig");
const Wave = @import("wave.zig").Wave;
const Window = @import("../windows.zig");
const GuiPositions = @import("guiPositions.zig");

pub fn window(demo: *Signals, gctx: *zgpu.GraphicsContext) void {
    Window.setNextWindow(gctx, GuiPositions.parameter);

    if (zgui.begin("Parameters", Window.window_flags)) {
        defer zgui.end();
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        zgui.text("Number random points", .{});
        if (zgui.sliderScalar("##nppc", u32, .{
            .v = &demo.random.pointsPerCycle,
            .min = 4,
            .max = 40,
        })) {
            const np = demo.random.pointsPerCycle;
            demo.input.pointsPerCycle = np;
            demo.random.createWave();
            demo.input.createComparisonWave(&demo.random);
        }
        waveInput(&demo.input, "1");

        const sum = demo.output.addWaveValues();
        zgui.text("Summation of Output wave = {d:.2}", .{sum});
    }
}

fn waveInput(wave: *Wave, comptime id: []const u8) void {
    const idStr = "##" ++ id;
    const sinId = idStr ++ "sin";
    const cosId = idStr ++ "cos";
    const nocId = idStr ++ "noc";

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });
    zgui.text("Wave Type", .{});

    _ = zgui.radioButtonStatePtr(sinId, .{
        .v = @ptrCast(*i32, &wave.waveType),
        .v_button = 0,
    });
    zgui.sameLine(.{});
    zgui.text("Sin", .{});
    zgui.sameLine(.{});
    _ = zgui.radioButtonStatePtr(cosId, .{
        .v = @ptrCast(*i32, &wave.waveType),
        .v_button = 1,
    });
    zgui.sameLine(.{});
    zgui.text("Cos", .{});

    zgui.text("Number of cycles", .{});
    _ = zgui.sliderScalar(nocId, u32, .{
        .v = &wave.cycles,
        .min = 1,
        .max = 20,
    });
}
