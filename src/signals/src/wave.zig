const std = @import("std");
const DemoState = @import("main.zig").DemoState;

const Point = struct {
    x: f32,
    y: f32,
};

pub const Wave = struct {
    xv: std.ArrayList(f32),
    yv: std.ArrayList(f32),
    params: Parameters,

    const Parameters = struct {
        num_cycles: u32 = 1,
        num_points_per_cycle: u32 = 10,
    };

    pub fn init(demo: *DemoState) Wave {
        return Wave{
            .xv = std.ArrayList(f32).init(demo.allocator),
            .yv = std.ArrayList(f32).init(demo.allocator),
            .params = Parameters{
                .num_cycles = demo.params.num_cycles,
                .num_points_per_cycle = demo.params.num_points_per_cycle,
            },
        };
    }

    pub fn deinit(self: Wave) void {
        self.xv.deinit();
        self.yv.deinit();
    }

    pub fn addPoint(self: *Wave, p: Point) !void {
        try self.xv.append(p.x);
        try self.yv.append(p.y);
    }

    pub fn createSinWave(demo: *DemoState) Wave {
        var wave = init(demo);
        var i: u32 = 0;
        while (i < wave.params.num_cycles) {
            createSinCycle(&wave, i);
            i += 1;
        }
        return wave;
    }
};

fn createSinCycle(wave: *Wave, cycle_offset: u32) void {
    var i: u32 = 0;
    var radians: f32 = 0;
    var num_increments = wave.params.num_points_per_cycle - 1;

    while (i < wave.params.num_points_per_cycle) {
        radians = @intToFloat(f32, i) / @intToFloat(f32, num_increments);
        radians *= std.math.tau;
        radians += (@intToFloat(f32, cycle_offset) * std.math.tau);
        const point = Point{
            .x = radians,
            .y = @sin(radians),
        };
        wave.addPoint(point) catch unreachable;
        i += 1;
    }
}
