const std = @import("std");
const zgui = @import("zgui");
const main = @import("main.zig");
const DemoState = main.DemoState;

fn Args(comptime T: type) type {
    return struct {
        label: [:0]const u8,
        id: [:0]const u8,
        min: T,
        max: T,
    };
}

pub fn setup() void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
}

pub fn displayFPS(demo: *DemoState) void {
    zgui.bulletText("{d:.1} fps", .{demo.gctx.stats.fps});
}

pub fn waveInput(demo: *DemoState) void {
    zgui.text("Number points per cycle", .{});
    if (zgui.sliderScalar("##nppc", u32, .{
        .v = &demo.input_one.params.num_points_per_cycle,
        .min = 4,
        .max = 40,
    })) {
        const np = demo.input_one.params.num_points_per_cycle;
        demo.input_two.params.num_points_per_cycle = np;
    }

    const nppc = demo.input_one.params.num_points_per_cycle;
    const num_points_one = demo.input_one.wave.xv.items.len;
    const num_points_two = demo.input_two.wave.xv.items.len;
    const max_num_points = @max(num_points_one, num_points_two);
    const max_shift_possible = @intCast(u32, max_num_points - nppc);

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });
    zgui.text("Number of cycles", .{});
    _ = zgui.sliderScalar("##1noc", u32, .{
        .v = &demo.input_one.params.num_cycles,
        .min = 1,
        .max = 20,
    });
    zgui.text("Shift", .{});
    _ = zgui.sliderScalar("##1shift", u32, .{
        .v = &demo.input_one.params.shift,
        .min = 0,
        .max = max_shift_possible,
    });

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });
    zgui.text("Number of cycles", .{});
    _ = zgui.sliderScalar("##2noc", u32, .{
        .v = &demo.input_two.params.num_cycles,
        .min = 1,
        .max = 20,
    });
    zgui.text("Shift", .{});
    _ = zgui.sliderScalar("##2shift", u32, .{
        .v = &demo.input_two.params.shift,
        .min = 0,
        .max = max_shift_possible,
    });

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });
}
