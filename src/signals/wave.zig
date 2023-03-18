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
        var wave = Wave.init(allocator);
        wave.createWave(params);
        return wave;
    }

    pub fn deinit(self: Wave) void {
        self.xv.deinit();
        self.yv.deinit();
    }

    pub fn clearWave(self: *Wave) void {
        self.xv.clearRetainingCapacity();
        self.yv.clearRetainingCapacity();
    }

    pub fn createWave(self: *Wave, params: InputParams) void {
        var point: u32 = 0;
        var radians: f32 = 0;
        const numCycles = @intToFloat(f32, params.num_cycles);
        const nppc = @intToFloat(f32, params.num_points_per_cycle);
        const num_points = @floatToInt(u32, nppc + ((nppc - 1) * (numCycles - 1)));
        while (point < num_points) {
            const current_point = @intToFloat(f32, point + params.shift);
            const point_num = current_point / (nppc - 1);
            radians = point_num * std.math.tau;
            self.xv.append(radians) catch unreachable;
            switch (params.waveType) {
                .sin => self.yv.append(@sin(radians)) catch unreachable,
                .cos => self.yv.append(@cos(radians)) catch unreachable,
            }
            point += 1;
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
        one: *const Wave,
        two: *const Wave,
    ) void {
        const oneStart = one.xv.items[0];
        const twoStart = two.xv.items[0];
        var first = one;
        var second = two;
        if (oneStart > twoStart) {
            first = two;
            second = one;
        }
        var secondIdx: u32 = 0;
        for (first.xv.items, 0..) |firstX, firstIdx| {
            if (secondIdx >= second.xv.items.len) {
                break;
            }
            const secondX = second.xv.items[secondIdx];
            const diff = @fabs(firstX - secondX);
            if (diff < 0.001) {
                const firstY = first.yv.items[firstIdx];
                const secondY = second.yv.items[secondIdx];
                self.xv.append(firstX) catch unreachable;
                self.yv.append(firstY * secondY) catch unreachable;
                secondIdx += 1;
            }
        }
    }
};
