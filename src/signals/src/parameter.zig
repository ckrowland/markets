pub const zgui = @import("zgui");
pub const main = @import("main.zig");

fn Args(comptime T: type) type {
    return struct {
        label: [:0]const u8,
        id: [:0]const u8,
        min: T,
        max: T,
    };
}

pub const Shift = Args(f32){
    .label = "Shift",
    .id = "##shift",
    .min = -100,
    .max = 100,
};

pub const NumPointsPerCycle = Args(u32){
    .label = "Number points per cycle",
    .id = "##nppc",
    .min = 4,
    .max = 200,
};

pub const NumCycles = Args(u32){
    .label = "Number of cycles",
    .id = "##noc",
    .min = 1,
    .max = 20,
};

pub fn setup() void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
}

pub fn displayFPS(demo: *main.DemoState) void {
    zgui.bulletText("{d:.1} fps", .{demo.gctx.stats.fps});
}

pub fn slider(comptime T: type, v: *T, comptime args: Args(T)) void {
    zgui.text(args.label, .{});
    _ = zgui.sliderScalar(args.id, T, .{ .v = v, .min = args.min, .max = args.max });
}
