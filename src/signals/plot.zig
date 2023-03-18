const zgui = @import("zgui");

pub fn setup(max_x: f32) void {
    setupAxes(max_x);
    setupMarkers();
}

fn setupAxes(max_x: f32) void {
    const setupAxis = .{ .label = "", .flags = .{}};
    zgui.plot.setupAxis(.x1, setupAxis);
    zgui.plot.setupAxis(.y1, .{
        .label = "",
        .flags = .{
            .auto_fit = true,
        },
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
