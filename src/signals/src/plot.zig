const zgui = @import("zgui");

const axisFlags = .{
    .auto_fit = true,
};

pub fn setup() void {
    setupAxes();
    setupMarkers();
}

fn setupAxes() void {
    const setupAxis = .{ .label = "", .flags = axisFlags };
    zgui.plot.setupAxis(.x1, setupAxis);
    zgui.plot.setupAxis(.y1, setupAxis);
}

fn setupMarkers() void {
    zgui.plot.pushStyleVar1i(.{
        .idx = zgui.plot.StyleVar.marker,
        .v = 1,
    });
    zgui.plot.pushStyleVar1f(.{
        .idx = zgui.plot.StyleVar.marker_size,
        .v = 10.0,
    });
}
