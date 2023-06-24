const std = @import("std");
const zgpu = @import("zgpu");
const zmath = @import("zmath");

// Camera Position
pub const POS_X: f32 = 0.0;
pub const POS_Y: f32 = 0.0;
pub const POS_Z: f32 = -3000.0;

// Grid limits for absolute positions (without aspect ratio)
pub const MAX_X: i32 = 1000;
pub const MIN_X: i32 = -1000;
pub const MAX_Y: i32 = 1000;
pub const MIN_Y: i32 = -1000;

// Viewport size relative to total window size
pub const VP_X_SIZE: f32 = 0.75;
pub const VP_Y_SIZE: f32 = 0.75;

pub fn getAspectRatio(gctx: *zgpu.GraphicsContext) f32 {
    const sd = gctx.swapchain_descriptor;
    return @floatFromInt(f32, sd.width) / @floatFromInt(f32, sd.height);
}

// Given a world position (with aspect), calculate grid position
pub fn getGridPosition(gctx: *zgpu.GraphicsContext, world_pos: [4]f32) zmath.F32x4 {
    const aspect = getAspectRatio(gctx);
    return .{
        world_pos[0] / aspect,
        world_pos[1],
        world_pos[2],
        world_pos[3],
    };
}

// Given a grid position, return a world position
pub fn getWorldPosition(gctx: *zgpu.GraphicsContext, grid_pos: [4]f32) [4]f32 {
    const aspect = getAspectRatio(gctx);
    return .{
        grid_pos[0] * aspect,
        grid_pos[1],
        grid_pos[2],
        grid_pos[3],
    };
}

pub fn getObjectToClipMat(gctx: *zgpu.GraphicsContext) zmath.Mat {
    const camWorldToView = zmath.lookAtLh(
        //eye position
        zmath.f32x4(POS_X, POS_Y, POS_Z, 0.0),

        //focus position
        zmath.f32x4(0.0, 0.0, 0.0, 0.0),

        //up direction
        zmath.f32x4(0.0, 1.0, 0.0, 0.0),
    );

    const camViewToClip = zmath.perspectiveFovLh(
        //fovy
        0.22 * std.math.pi,

        //aspect
        getAspectRatio(gctx),

        //near
        0.01,

        //far
        3000.0,
    );
    const camWorldToClip = zmath.mul(camWorldToView, camViewToClip);
    // return zmath.transpose(camWorldToClip);
    return camWorldToClip;
}
