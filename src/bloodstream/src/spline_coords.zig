const std = @import("std");
const array = std.ArrayList;
const Simulation = @import("simulation.zig");

pub const SplinePointCoords = struct {
    start: [2]f32,
    delta: [2]f32,
};

pub fn topLeftHeart(sim: *Simulation) array(SplinePointCoords) {
    var points = array(SplinePointCoords).init(sim.allocator);
    const cx = sim.coordinate_size.center_x;
    const cy = sim.coordinate_size.center_y;

    var heart_width: f32 = 300;
    var half_heart_width = heart_width / 2;
    var heart_height: f32 = 300;

    const beat_y: f32 = 100;

    points.appendSlice(&.{
        .{
            .start = [2]f32{cx - 200 - half_heart_width, cy - 100},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - half_heart_width, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
        },
        .{
            .start = [2]f32{cx, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
        },
        .{
            .start = [2]f32{cx + 200, cy - 100},
            .delta = [2]f32{0, 0},
        },
    }) catch unreachable;
    return points;
}

pub fn topRightHeart(sim: *Simulation) array(SplinePointCoords) {
    var points = array(SplinePointCoords).init(sim.allocator);
    const cx = sim.coordinate_size.center_x;
    const cy = sim.coordinate_size.center_y;

    var heart_width: f32 = 300;
    var half_heart_width = heart_width / 2;
    var heart_height: f32 = 300;

    const beat_y: f32 = 100;

    points.appendSlice(&.{
        .{
            .start = [2]f32{cx - 200, cy - 100},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
        },
        .{
            .start = [2]f32{cx + half_heart_width, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
        },
        .{
            .start = [2]f32{cx + 200 + half_heart_width, cy - 100},
            .delta = [2]f32{0, 0},
        },
    }) catch unreachable;
    return points;
}

pub fn bottomHeart(sim: *Simulation) array(SplinePointCoords) {
    var points = array(SplinePointCoords).init(sim.allocator);
    const cx = sim.coordinate_size.center_x;
    const cy = sim.coordinate_size.center_y;

    var heart_width: f32 = 300;
    var heart_height: f32 = 300;

    const beat_x: f32 = 100;
    const beat_y: f32 = 100;

    points.appendSlice(&.{
        .{
            .start = [2]f32{cx, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx + heart_width, cy + heart_height},
            .delta = [2]f32{-beat_x, -beat_y},
        },
        .{
            .start = [2]f32{cx + heart_width, cy},
            .delta = [2]f32{-beat_x, 0},
        },
        .{
            .start = [2]f32{cx, cy - heart_height},
            .delta = [2]f32{0, beat_y},
        },
        .{
            .start = [2]f32{cx - heart_width, cy},
            .delta = [2]f32{beat_x, 0},
        },
        .{
            .start = [2]f32{cx - heart_width, cy + heart_height},
            .delta = [2]f32{beat_x, -beat_y},
        },
        .{
            .start = [2]f32{cx, cy},
            .delta = [2]f32{0, 0},
        },
    }) catch unreachable;
    return points;
}

pub fn outerBloodstream(sim: *Simulation) array(SplinePointCoords) {
    var points = array(SplinePointCoords).init(sim.allocator);
    const cx = sim.coordinate_size.center_x;
    const cy = sim.coordinate_size.center_y;

    var heart_width: f32 = 300;
    var half_heart_width = heart_width / 2;
    var heart_height: f32 = 300;

    const beat_y: f32 = 100;

    var bloodstream_width: f32 = 1000;
    var bloodstream_height: f32 = 800;

    const outer_points = [_]SplinePointCoords{
        .{
            .start = [2]f32{cx, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx + half_heart_width, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
        },
        .{
            .start = [2]f32{cx + bloodstream_width, cy + bloodstream_height},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx + bloodstream_width, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx, cy - bloodstream_height},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - bloodstream_width, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - bloodstream_width, cy + bloodstream_height},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - half_heart_width, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
        },
        .{
            .start = [2]f32{cx, cy},
            .delta = [2]f32{0, 0},
        },
    };
    points.appendSlice(outer_points[0..]) catch unreachable;
    return points;
}

