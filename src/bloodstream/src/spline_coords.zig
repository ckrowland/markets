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

    var points = [4][2]f32{
        heart[4].start,
        heart[5].start,
        heart[6].start,
        heart[7].start,
    };
    const left_opening = Spline.getSplinePoint(0.8, points);

    points = [4][2]f32{
        heart[5].start,
        heart[6].start,
        heart[7].start,
        heart[8].start,
    };
    const left_closing = Spline.getSplinePoint(0.2, points);

    points = [4][2]f32{
        heart[0].start,
        heart[1].start,
        heart[2].start,
        heart[3].start,
    };
    const right_opening = Spline.getSplinePoint(0.8, points);

    points = [4][2]f32{
        heart[1].start,
        heart[2].start,
        heart[3].start,
        heart[4].start,
    };
    const right_closing = Spline.getSplinePoint(0.2, points);

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
