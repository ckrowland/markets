const std = @import("std");
const zgpu = @import("zgpu");
const zmath = @import("zmath");

// Camera Settings
pub const POS: [3]f32 = .{ 0.0, 0.0, -3000.0 };
pub const FOCUS: [3]f32 = .{ 0.0, 0.0, 0.0 };
pub const UP: [4]f32 = .{ 0.0, 1.0, 0.0, 0.0 };

pub const FOV_Y: f32 = 0.22 * std.math.pi;
pub const NEAR_PLANE: f32 = 0.01;
pub const FAR_PLANE: f32 = 3000.0;

// Grid limits for grid positions (without aspect ratio)
pub const MAX_X: i32 = 1000;
pub const MIN_X: i32 = -1000;
pub const MAX_Y: i32 = 1000;
pub const MIN_Y: i32 = -1000;
pub const TOTAL_X: i32 = 2000;
pub const TOTAL_Y: i32 = 2000;

// Viewport size relative to total window size
pub const VP_X_SIZE: f32 = 0.8;
pub const VP_Y_SIZE: f32 = 0.8;

pub fn getViewportPixelSize(gctx: *zgpu.GraphicsContext) [2]f32 {
    return .{
        @as(f32, @floatFromInt(gctx.swapchain_descriptor.width)) * VP_X_SIZE,
        @as(f32, @floatFromInt(gctx.swapchain_descriptor.height)) * VP_Y_SIZE,
    };
}

pub fn getAspectRatio(gctx: *zgpu.GraphicsContext) f32 {
    const sd = gctx.swapchain_descriptor;
    return @as(f32, @floatFromInt(sd.width)) / @as(f32, @floatFromInt(sd.height));
}

// Grid Position: (MIN_X -> MAX_X, MIN_Y -> MAX_Y) never changes, pure grid
// World Position: Grid Position with x axis multiplied by aspect ratio
// Pixel Position: World Position as seen from the camera.
//                 Pixel coordinates rendered.

// Given a world position (grid position with aspect), return grid position
pub fn getGridFromWorld(gctx: *zgpu.GraphicsContext, world_pos: [4]f32) [4]f32 {
    const aspect = getAspectRatio(gctx);
    return .{ world_pos[0] / aspect, world_pos[1], world_pos[2], world_pos[3] };
}

// Given a grid position, return a world position
pub fn getWorldPosition(gctx: *zgpu.GraphicsContext, grid_pos: [4]f32) [4]f32 {
    const aspect = getAspectRatio(gctx);
    return .{ grid_pos[0] * aspect, grid_pos[1], grid_pos[2], grid_pos[3] };
}

// Given a grid position, return a pixel position
pub fn getPixelPosition(gctx: *zgpu.GraphicsContext, g_pos: [2]f32) [2]f32 {
    const grid_pos = .{ g_pos[0], g_pos[1], 0, 1 };
    const world_pos = zmath.loadArr4(getWorldPosition(gctx, grid_pos));
    const camera_pos = zmath.mul(world_pos, getObjectToClipMat(gctx));
    const rel_pos = [4]f32{ camera_pos[0] / -POS[2], camera_pos[1] / -POS[2], 0, 1 };
    const viewport_size = getViewportPixelSize(gctx);
    const width = @as(f32, @floatFromInt(gctx.swapchain_descriptor.width));
    const xOffset = width - viewport_size[0];

    const cursor_in_vp_x = ((rel_pos[0] + 1) * viewport_size[0]) / 2;
    const cursor_in_vp_y = ((-rel_pos[1] + 1) * viewport_size[1]) / 2;
    return .{ cursor_in_vp_x + xOffset, cursor_in_vp_y };
}

// Given a pixel position, return a grid position
pub fn getGridPosition(gctx: *zgpu.GraphicsContext, p_pos: [4]f32) [4]i32 {
    const viewport_size = getViewportPixelSize(gctx);
    const width = @as(f32, @floatFromInt(gctx.swapchain_descriptor.width));
    const xOffset = width - viewport_size[0];

    const rel_pos_x = (((p_pos[0] - xOffset) * 2) / viewport_size[0]) - 1;
    const rel_pos_y = ((p_pos[1] * 2) / viewport_size[1]) - 1;

    const camera_pos = zmath.loadArr4(.{
        rel_pos_x * -POS[2],
        rel_pos_y * POS[2],
        0,
        1,
    });
    const inverse_mat = zmath.inverse(getObjectToClipMat(gctx));
    const world_pos = zmath.mul(camera_pos, inverse_mat);
    return getGridFromWorld(gctx, .{ world_pos[0], world_pos[1], 0, 1 });
}

pub fn getObjectToClipMat(gctx: *zgpu.GraphicsContext) zmath.Mat {
    const camWorldToView = zmath.lookAtLh(
        zmath.loadArr3(POS),
        zmath.loadArr3(FOCUS),
        zmath.loadArr4(UP),
    );
    const camViewToClip = zmath.perspectiveFovLh(
        FOV_Y,
        getAspectRatio(gctx),
        NEAR_PLANE,
        FAR_PLANE,
    );

    return zmath.mul(camWorldToView, camViewToClip);
}
