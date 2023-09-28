const std = @import("std");
const zgpu = @import("zgpu");
const zmath = @import("zmath");

// Camera Position
pub const POS_X: f32 = 500.0;
pub const POS_Y: f32 = 500.0;
pub const POS_Z: f32 = -500.0;

// Grid limits for absolute positions (without aspect ratio)
pub const MAX_X: i32 = 1000;
pub const MIN_X: i32 = -1000;
pub const MAX_Y: i32 = 1000;
pub const MIN_Y: i32 = -1000;
pub const TOTAL_X: i32 = 2000;
pub const TOTAL_Y: i32 = 2000;

// Viewport size relative to total window size
pub const VP_X_SIZE: f32 = 0.75;
pub const VP_Y_SIZE: f32 = 0.75;

pub fn getViewportPixelSize(gctx: *zgpu.GraphicsContext) [2]f32 {
    const width = @as(f32, @floatFromInt(gctx.swapchain_descriptor.width));
    const height = @as(f32, @floatFromInt(gctx.swapchain_descriptor.height));
    return .{ width * VP_X_SIZE, height * VP_Y_SIZE };
}

pub fn getAspectRatio(gctx: *zgpu.GraphicsContext) f32 {
    const sd = gctx.swapchain_descriptor;
    return @as(f32, @floatFromInt(sd.width)) / @as(f32, @floatFromInt(sd.height));
}

// Given a world position (grid position with aspect), return grid position
pub fn getGridPosition(gctx: *zgpu.GraphicsContext, world_pos: [2]f32) [2]i32 {
    const aspect = getAspectRatio(gctx);
    return .{
        @as(i32, @intFromFloat(world_pos[0] / aspect)),
        @as(i32, @intFromFloat(world_pos[1])),
        // world_pos[2],
        // world_pos[3],
    };
}

// Given a grid position, return a world position
pub fn getWorldPosition(gctx: *zgpu.GraphicsContext, grid_pos: [4]i32) [4]f32 {
    const aspect = getAspectRatio(gctx);
    return .{
        @as(f32, @floatFromInt(grid_pos[0])) * aspect,
        @as(f32, @floatFromInt(grid_pos[1])),
        @as(f32, @floatFromInt(grid_pos[2])),
        @as(f32, @floatFromInt(grid_pos[3])),
    };
}

// Given a grid position, return a pixel position
pub fn getPixelPosition(gctx: *zgpu.GraphicsContext, g_pos: [2]i32) [2]f32 {
    const grid_pos = .{ g_pos[0], g_pos[1], 0, 1 };
    const world_pos = zmath.loadArr4(getWorldPosition(gctx, grid_pos));
    const camera_pos = zmath.mul(world_pos, getObjectToClipMat(gctx));
    const rel_pos = [4]f32{ camera_pos[0] / -POS_Z, camera_pos[1] / -POS_Z, 0, 1 };

    const viewport_size = getViewportPixelSize(gctx);
    const width = @as(f32, @floatFromInt(gctx.swapchain_descriptor.width));
    const xOffset = width - viewport_size[0];
    const content_scale = gctx.window.getContentScale();

    const cursor_in_vp_x = ((rel_pos[0] + 1) * viewport_size[0]) / (2 * content_scale[0]);
    const cursor_in_vp_y = ((-rel_pos[1] + 1) * viewport_size[1]) / (2 * content_scale[1]);
    const screen_coords = [2]f32{ cursor_in_vp_x + (xOffset / 2), cursor_in_vp_y };
    return .{ screen_coords[0] * content_scale[0], screen_coords[1] * content_scale[1] };
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
