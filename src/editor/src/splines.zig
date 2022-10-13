const std = @import("std");
const math = std.math;
const random = std.crypto.random;
const array = std.ArrayList;
const zgpu = @import("zgpu");
const wgpu = zgpu.wgpu;
const Simulation = @import("simulation.zig");
const main = @import("editor.zig");
const Vertex = main.Vertex;
const DemoState = main.DemoState;
const SplineCoords = @import("spline_coords.zig");
const SplinePointCoords = SplineCoords.SplinePointCoords;

pub const SplinePoint = struct {
    pub const zeros = [4]f32{ 0, 0, 0, 0 };

    color: [4]f32 = zeros,
    start_pos: [4]f32 = zeros,
    current_pos: [4]f32 = zeros,
    end_pos: [4]f32 = zeros,
    step_size: [4]f32 = zeros,
    radius: f32 = 0,
    to_start: u32 = 0,
    spline_id: u32 = 0,
    point_id: u32 = 0,
    len: u32 = 0,
    _padding: u64 = 0,
};


pub fn createTestSpline(self: *Simulation, ox: f32, oy: f32) void {
    const test_points = SplineCoords.getTestPoints(self, ox, oy);
    var splines = array(array(SplinePointCoords)).init(self.allocator);
    splines.append(test_points) catch unreachable;

    for (splines.items) |spline, idx| {
        for (spline.items) |p, i| {
            const len = @intCast(u32, spline.items.len);
            const color = [4]f32 { 0, 1, 1, 0 };
            const num_steps = 100;
            const radius = 5;

            const x = p.start[0];
            const y = p.start[1];
            const dx = p.delta[0];
            const dy = p.delta[1];

            const x_step_size = -dx / @intToFloat(f32, num_steps);
            const y_step_size = -dy / @intToFloat(f32, num_steps);
            const step_size = [4]f32{ x_step_size, y_step_size, 0, 0 };
            const start = [4]f32{ x, y, 0, 0 };
            const end = [4]f32{ x + dx, y + dy, 0, 0 };

            self.asplines.append(.{
                .color = color,
                .start_pos = start,
                .current_pos = start,
                .end_pos = end,
                .step_size = step_size,
                .radius = radius,
                .to_start = 0,
                .spline_id = @intCast(u32, idx),
                .point_id = @intCast(u32, i),
                .len = len,
                ._padding = 0,
            }) catch unreachable;
        }
    }
    test_points.deinit();
    splines.deinit();
}

pub fn createSplines(self: *Simulation) void {
    const color = [4]f32 { 0, 1, 1, 0 };
    const num_steps = 100;
    const radius = 5;

    const outer_bloodstream = SplineCoords.outerBloodstream(self);
    const inner_bloodstream = SplineCoords.innerBloodstream(self);
    const top_left_heart = SplineCoords.topLeftHeart(self);
    const top_right_heart = SplineCoords.topRightHeart(self);
    const bottom_heart = SplineCoords.bottomHeart(self);

    var splines = array(array(SplinePointCoords)).init(self.allocator);
    splines.append(outer_bloodstream) catch unreachable;
    splines.append(inner_bloodstream) catch unreachable;
    splines.append(top_left_heart) catch unreachable;
    splines.append(top_right_heart) catch unreachable;
    splines.append(bottom_heart) catch unreachable;

    for (splines.items) |spline, idx| {
        for (spline.items) |p, i| {
            const len = @intCast(u32, spline.items.len);

            const x = p.start[0];
            const y = p.start[1];
            const dx = p.delta[0];
            const dy = p.delta[1];

            const x_step_size = -dx / @intToFloat(f32, num_steps);
            const y_step_size = -dy / @intToFloat(f32, num_steps);
            const step_size = [4]f32{ x_step_size, y_step_size, 0, 0 };
            const start = [4]f32{ x, y, 0, 0 };
            const end = [4]f32{ x + dx, y + dy, 0, 0 };

            self.asplines.append(.{
                .color = color,
                .start_pos = start,
                .current_pos = start,
                .end_pos = end,
                .step_size = step_size,
                .radius = radius,
                .to_start = 0,
                .spline_id = @intCast(u32, idx),
                .point_id = @intCast(u32, i),
                .len = len,
                ._padding = 0,
            }) catch unreachable;
        }
    }

    outer_bloodstream.deinit();
    inner_bloodstream.deinit();
    top_left_heart.deinit();
    top_right_heart.deinit();
    bottom_heart.deinit();
    splines.deinit();
}


pub fn getSplinePoints(ref_points: []SplinePoint, num_points: u32) []SplinePoint {
    std.debug.assert(ref_points.len == 4);

    const max_num_points = 10000;
    var points: [max_num_points]SplinePoint = undefined;


    const p0x = ref_points[0].current_pos[0];
    const p1x = ref_points[1].current_pos[0];
    const p2x = ref_points[2].current_pos[0];
    const p3x = ref_points[3].current_pos[0];

    const p0y = ref_points[0].current_pos[1];
    const p1y = ref_points[1].current_pos[1];
    const p2y = ref_points[2].current_pos[1];
    const p3y = ref_points[3].current_pos[1];

    var i: u32 = 0;
    while (i < num_points) {
        const t = @intToFloat(f32, i / num_points);
        const tt = t * t;
        const ttt = tt * t;

        const q1 = -ttt + (2 * tt) - t;
        const q2 = (3 * ttt) - (5 * tt) + 2;
        const q3 = (-3 * ttt) + (4 * tt) + t;
        const q4 = ttt - tt;


        var tx = (p0x * q1) + (p1x * q2) + (p2x * q3) + (p3x * q4);
        var ty = (p0y * q1) + (p1y * q2) + (p2y * q3) + (p3y * q4);

        tx *= 0.5;
        ty *= 0.5;

        points[i] = SplinePoint{
            .color = ref_points[1].color,
            .current_pos = [4]f32{tx, ty, 0, 0 },
            .radius = ref_points[1].radius,
        };
        i += 1;
    }

    return points[0..i+1];
}

pub fn createSplinePointBuffer(gctx: *zgpu.GraphicsContext, splines: *array(SplinePoint)) zgpu.BufferHandle {
    const splines_point_buffer = gctx.createBuffer(.{
        .usage = .{ 
            .copy_dst = true, 
            .copy_src = true, 
            .vertex = true,
            .storage = true,
        },
        .size = splines.items.len * @sizeOf(SplinePoint),
    });
    
    const cloned = splines.clone() catch unreachable;
    defer cloned.deinit();
    for (cloned.items) |_, i| {
        cloned.items[i].radius += 15;
    }

    gctx.queue.writeBuffer(gctx.lookupResource(splines_point_buffer).?, 0, SplinePoint, cloned.items[0..]);
    return splines_point_buffer;
}

pub fn createSplinesBuffer(gctx: *zgpu.GraphicsContext, points: array(SplinePoint), allocator: std.mem.Allocator) zgpu.BufferHandle {
    const num_spline_points = 1000;
    const num_points = points.items.len * (num_spline_points + 1);
    const splines_point_buffer = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .vertex = true, .storage = true },
        .size = num_points * @sizeOf(SplinePoint),
    });

    var point_data = array(SplinePoint).init(allocator);
    for (points.items) |sp, idx| {
        const first_point = sp.point_id == 0;
        const second_point = sp.point_id == 1;
        const last_point = sp.point_id == sp.len - 1;
        if (first_point or second_point or last_point) {
            point_data.appendNTimes(
                SplinePoint{},
                num_spline_points
            ) catch unreachable;
            continue;
        }

        const ref_points = points.items[idx-2..idx+2];
        const position = getSplinePoints(ref_points, num_spline_points);
        point_data.appendSlice(position) catch unreachable;
    }
    gctx.queue.writeBuffer(gctx.lookupResource(splines_point_buffer).?, 0, SplinePoint, point_data.items[0..]);
    point_data.deinit();
    return splines_point_buffer;
}

pub fn createAnimatedSplineComputePipeline(gctx: *zgpu.GraphicsContext, pipeline_layout: zgpu.PipelineLayoutHandle) zgpu.ComputePipelineHandle {
    const common_cs = @embedFile("shaders/compute/common.wgsl");
    const cs = common_cs ++ @embedFile("shaders/compute/spline_animation.wgsl");
    const cs_module = zgpu.createWgslShaderModule(gctx.device, cs, "cs");
    defer cs_module.release();

    const pipeline_descriptor = wgpu.ComputePipelineDescriptor{
        .compute = wgpu.ProgrammableStageDescriptor{
            .module = cs_module,
            .entry_point = "main",
        },
    };

    return gctx.createComputePipeline(pipeline_layout, pipeline_descriptor);
}
