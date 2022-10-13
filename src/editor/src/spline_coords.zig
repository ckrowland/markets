const std = @import("std");
const array = std.ArrayList;
const Simulation = @import("simulation.zig");
const Spline = @import("splines.zig");

pub const SplinePointCoords = struct {
    start: [2]f32,
    delta: [2]f32,
};

pub fn getTestPoints(sim: *Simulation, ox: f32, oy: f32) array(SplinePointCoords) {
    var points_array = array(SplinePointCoords).init(sim.allocator);
    const cx = sim.coordinate_size.center_x;
    const cy = sim.coordinate_size.center_y;

    points_array.appendSlice(&.{
        .{
            .start = [2]f32{cx - ox, cy - oy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - ox, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx + ox, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx + ox, cy - oy},
            .delta = [2]f32{0, 0},
        },
    }) catch unreachable;

    return points_array;
}

fn getHeartPoints(sim: *Simulation) [9]SplinePointCoords {
    const cx = sim.coordinate_size.center_x;
    const cy = sim.coordinate_size.center_y;

    var heart_width: f32 = 300;
    var heart_height: f32 = 300;

    const beat_x: f32 = 100;
    const beat_y: f32 = 100;
    
    return [9]SplinePointCoords{
        .{
            .start = [2]f32{cx, cy - heart_height},
            .delta = [2]f32{0, beat_y},
        },
        .{
            .start = [2]f32{cx, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
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
            .start = [2]f32{cx, cy + heart_height},
            .delta = [2]f32{0, -beat_y},
        },
        .{
            .start = [2]f32{cx, cy - heart_height},
            .delta = [2]f32{0, beat_y},
        }
    };
}

fn getBloodstreamCoords(sim: *Simulation, width: f32, height: f32) [9]SplinePointCoords {
    const cx = sim.coordinate_size.center_x;
    const cy = sim.coordinate_size.center_y;
    const heart_width = 300;
    const heart_height = 300;
    const beat_x = 100;
    const beat_y = 100;

    return [9]SplinePointCoords{
        .{
            .start = [2]f32{cx, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx + heart_width, cy + heart_height},
            .delta = [2]f32{-beat_x, -beat_y},
        },
        .{
            .start = [2]f32{cx + width, cy + height},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx + width, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx, cy - height},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - width, cy},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - width, cy + height},
            .delta = [2]f32{0, 0},
        },
        .{
            .start = [2]f32{cx - heart_width, cy + heart_height},
            .delta = [2]f32{beat_x, -beat_y},
        },
        .{
            .start = [2]f32{cx, cy},
            .delta = [2]f32{0, 0},
        },
    };
}
fn getHeartOpenings(sim: *Simulation) [4]SplinePointCoords {
    const heart = getHeartPoints(sim);

    const left_opening = getSplinePoint(0.8, heart[4..8].*);
    const left_closing = getSplinePoint(0.2, heart[5..9].*);
    const right_opening = getSplinePoint(0.8, heart[0..4].*);
    const right_closing = getSplinePoint(0.2, heart[1..5].*);

    return [4]SplinePointCoords{
        .{
            .start = left_opening,
            .delta = heart[6].delta,
        },
        .{
            .start = left_closing,
            .delta = heart[6].delta,
        },
        .{
            .start = right_opening,
            .delta = heart[2].delta,
        },
        .{
            .start = right_closing,
            .delta = heart[2].delta,
        },
    };
}

pub fn topLeftHeart(sim: *Simulation) array(SplinePointCoords) {
    var points_array = array(SplinePointCoords).init(sim.allocator);

    const heart = getHeartPoints(sim);
    const opening = getHeartOpenings(sim);

    points_array.appendSlice(&.{
        heart[5],
        opening[1],
        heart[7],
        heart[8],
    }) catch unreachable;

    return points_array;
}

pub fn topRightHeart(sim: *Simulation) array(SplinePointCoords) {
    var points_array = array(SplinePointCoords).init(sim.allocator);

    const heart = getHeartPoints(sim);
    const opening = getHeartOpenings(sim);

    points_array.appendSlice(&.{
        heart[0],
        heart[1],
        opening[2],
        heart[3],
    }) catch unreachable;
    return points_array;
}

pub fn bottomHeart(sim: *Simulation) array(SplinePointCoords) {
    var points_array = array(SplinePointCoords).init(sim.allocator);

    const heart = getHeartPoints(sim);
    const opening = getHeartOpenings(sim);

    points_array.appendSlice(&.{
        heart[7],
        opening[0],
        heart[5],
        heart[4],
        heart[3],
        opening[3],
        heart[1],
    }) catch unreachable;
    return points_array;
}

pub fn innerBloodstream(sim: *Simulation) array(SplinePointCoords) {
    var points = array(SplinePointCoords).init(sim.allocator);
    var bloodstream = getBloodstreamCoords(sim, 800, 600);

    const opening = getHeartOpenings(sim);
    bloodstream[1] = opening[3];
    bloodstream[7] = opening[0];

    points.appendSlice(bloodstream[0..]) catch unreachable;
    return points;
}

pub fn outerBloodstream(sim: *Simulation) array(SplinePointCoords) {
    var points = array(SplinePointCoords).init(sim.allocator);
    var bloodstream = getBloodstreamCoords(sim, 1000, 800);

    const opening = getHeartOpenings(sim);
    bloodstream[1] = opening[2];
    bloodstream[7] = opening[1];

    points.appendSlice(bloodstream[0..]) catch unreachable;
    return points;
}

pub fn getSplinePoint(i: f32, points: [4]SplinePointCoords) [2]f32 {
    const t = i;
    const tt = t * t;
    const ttt = tt * t;

    const q1 = -ttt + (2 * tt) - t;
    const q2 = (3 * ttt) - (5 * tt) + 2;
    const q3 = (-3 * ttt) + (4 * tt) + t;
    const q4 = ttt - tt;

    const p0x = points[0].start[0];
    const p1x = points[1].start[0];
    const p2x = points[2].start[0];
    const p3x = points[3].start[0];

    const p0y = points[0].start[1];
    const p1y = points[1].start[1];
    const p2y = points[2].start[1];
    const p3y = points[3].start[1];

    var tx = (p0x * q1) + (p1x * q2) + (p2x * q3) + (p3x * q4);
    var ty = (p0y * q1) + (p1y * q2) + (p2y * q3) + (p3y * q4);

    tx *= 0.5;
    ty *= 0.5;

    return [2]f32{ tx, ty };
}
