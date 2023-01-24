const std = @import("std");
const main = @import("main.zig");
const DemoState = main.DemoState;
const InputParams = main.Parameters;

pub const Wave = struct {
    xv: std.ArrayList(f32),
    yv: std.ArrayList(f32),

    pub fn init(allocator: std.mem.Allocator) Wave {
        return Wave{
            .xv = std.ArrayList(f32).init(allocator),
            .yv = std.ArrayList(f32).init(allocator),
        };
    }

    pub fn initAndGenerate(
        allocator: std.mem.Allocator,
        params: InputParams,
    ) Wave {
        var xv = std.ArrayList(f32).init(allocator);
        var yv = std.ArrayList(f32).init(allocator);

        createSinCycle(&xv, &yv, params, 0);

        var i: u32 = 1;
        while (i < params.num_cycles) {
            createSinCycle(&xv, &yv, params, i);
            i += 1;
        }
        return Wave{
            .xv = xv,
            .yv = yv,
        };
    }

    pub fn deinit(self: Wave) void {
        self.xv.deinit();
        self.yv.deinit();
    }

    pub fn clearWave(self: *Wave) void {
        self.xv.clearRetainingCapacity();
        self.yv.clearRetainingCapacity();
    }

    // Put outside Wave struct
    pub fn createSinWave(self: *Wave, params: InputParams) void {
        createSinCycle(&self.xv, &self.yv, params, 0);

        var i: u32 = 1;
        while (i < params.num_cycles) {
            createSinCycle(&self.xv, &self.yv, params, i);
            i += 1;
        }
    }

    pub fn isInRange(self: *const Wave, x: f32) bool {
        const start = self.xv.items[0];
        const end = self.xv.items[self.xv.items.len - 1];
        if (x < start) return false;
        if (x > end) return false;
        return true;
    }

    pub fn getValue(self: *const Wave, x: f32) f32 {
        const margin = 0.05;
        for (self.xv.items) |value| {
            if (value > x - margin and value < x + margin) {
                return value;
            }
        }
        unreachable;
    }

    pub fn getLastPointX(self: *const Wave) f32 {
        return self.xv.items[self.xv.items.len - 1];
    }

    pub fn multiplyWaves(
        self: *Wave,
        first: *const Wave,
        second: *const Wave,
    ) void {
        var second_idx: u32 = 0;
        for (first.xv.items) |first_x, first_idx| {
            if (second_idx >= second.xv.items.len) {
                break;
            }
            const second_x = second.xv.items[second_idx];
            const diff = @fabs(first_x - second_x);
            if (diff < 0.001) {
                const first_y = first.yv.items[first_idx];
                const second_y = second.yv.items[second_idx];
                self.xv.append(first_x) catch unreachable;
                self.yv.append(first_y * second_y) catch unreachable;
                second_idx += 1;
            }
        }
    }
};

fn createSinCycle(
    xv: *std.ArrayList(f32),
    yv: *std.ArrayList(f32),
    params: InputParams,
    cycle_offset: u32,
) void {
    var i: u32 = 1;
    if (cycle_offset == 0) i = 0;

    var radians: f32 = 0;
    var num_increments = @intToFloat(f32, params.num_points_per_cycle - 1);

    const cycle_num = @intToFloat(f32, cycle_offset);
    while (i < params.num_points_per_cycle) {
        const current_point = @intToFloat(f32, i + params.shift);
        const point_num = current_point / num_increments;

        radians = (cycle_num + point_num) * std.math.tau;
        xv.append(radians) catch unreachable;
        yv.append(@sin(radians)) catch unreachable;
        i += 1;
    }
}
