const std = @import("std");
const zgui = @import("zgui");
const Signals = @import("main.zig");

fn Args(comptime T: type) type {
    return struct {
        label: [:0]const u8,
        id: [:0]const u8,
        min: T,
        max: T,
    };
}

pub fn waveInput(demo: *Signals) void {
    const nppc = demo.input_one.params.num_points_per_cycle;
    const num_points_one = demo.input_one.wave.xv.items.len;
    const num_points_two = demo.input_two.wave.xv.items.len;
    const max_num_points = @max(num_points_one, num_points_two);
    const max_shift_possible = @intCast(u32, max_num_points - nppc);

    zgui.text("Number points per cycle", .{});
    if (zgui.sliderScalar("##nppc", u32, .{
        .v = &demo.input_one.params.num_points_per_cycle,
        .min = 4,
        .max = 40,
    })) {
        const np = demo.input_one.params.num_points_per_cycle;
        demo.input_two.params.num_points_per_cycle = np;
    }

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });
    zgui.text("Wave Type", .{});
    _ = zgui.combo("Combo 1", .{
        .current_item = @ptrCast(*i32, &demo.input_one.params.waveType),
        .items_separated_by_zeros = "sin\x00cos\x00",
    });
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
    zgui.text("Wave Type", .{});
    _ = zgui.combo("Combo 2", .{
        .current_item = @ptrCast(*i32, &demo.input_two.params.waveType),
        .items_separated_by_zeros = "sin\x00cos\x00",
    });
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
