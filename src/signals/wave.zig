const std = @import("std");
const main = @import("main.zig");
const DemoState = main.DemoState;
const InputParams = main.Parameters;
const Signal = main.Signal;

// Should I have different structs for random wave, sin/cos wave, result wave?
// Then have a super struct of a fourier transform with a random, input and resulting
// frequency graph struct?

pub const Wave = struct {
    id: u32,
    pointsPerCycle: u32 = 10,
    cycles: u32 = 6,
    waveType: waveType = .sin,
    xv: std.ArrayList(f32),
    yv: std.ArrayList(f32),

    pub const waveType = enum(i32) {
        sin,
        cos,
        random,
        result,
    };

    pub fn init(allocator: std.mem.Allocator, id: u32, wType: waveType) Wave {
        return Wave{
            .id = id,
            .waveType = wType,
            .xv = std.ArrayList(f32).init(allocator),
            .yv = std.ArrayList(f32).init(allocator),
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

    pub fn calculateNumPoints(self: *Wave) u32 {
        const firstCycle = self.pointsPerCycle;
        const remainingCycles = (self.pointsPerCycle - 1) * (self.cycles - 1);
        return firstCycle + remainingCycles;
    }

    pub fn createWave(self: *Wave) void {
        self.clearWave();
        var point: u32 = 0;
        var radians: f32 = 0;
        const numPoints = self.calculateNumPoints();
        const numIncrements = @floatFromInt(f32, self.pointsPerCycle - 1);
        while (point < numPoints) : (point += 1) {
            const fPoint = @floatFromInt(f32, point);
            const point_num = fPoint / numIncrements;
            radians = point_num * std.math.tau;
            self.xv.append(radians) catch unreachable;

            switch (self.waveType) {
                .sin => self.yv.append(@sin(radians)) catch unreachable,
                .cos => self.yv.append(@cos(radians)) catch unreachable,
                .random => {
                    const r = std.crypto.random.float(f32) * 2 - 1;
                    self.yv.append(r) catch unreachable;
                },
                .result => unreachable,
            }
        }
    }

    pub fn createComparisonWave(self: *Wave, random: *Wave) void {
        self.clearWave();
        const endRadian = random.getLastPointX();
        const fNumCycles = @floatFromInt(f32, self.cycles);

        for (random.xv.items) |xv| {
            const percentToEndRadian = xv / (endRadian / fNumCycles);
            const twoPiRadian = percentToEndRadian * std.math.tau;
            self.xv.append(xv) catch unreachable;
            switch (self.waveType) {
                .sin => self.yv.append(@sin(twoPiRadian)) catch unreachable,
                .cos => self.yv.append(@cos(twoPiRadian)) catch unreachable,
                .random => unreachable,
                .result => unreachable,
            }
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
        self.clearWave();
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

    pub fn addWaveValues(self: *Wave) f32 {
        var sum: f32 = 0;
        for (self.yv.items) |value| {
            sum += value;
        }
        return sum;
    }
};
