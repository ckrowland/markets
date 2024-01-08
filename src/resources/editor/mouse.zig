const std = @import("std");
const zglfw = @import("zglfw");
const zm = @import("zmath");
const zgpu = @import("zgpu");
const Camera = @import("../camera.zig");
const zems = @import("zems");
const emscripten = zems.is_emscripten;

pub const MouseButton = struct {
    name: [:0]const u8 = "Primary",
    button: zglfw.MouseButton = zglfw.MouseButton.left,
    state: bool = false,
    previousState: bool = false,
    grid_pos: [2]i32 = .{ 0, 0 },
    pixel_pos: [2]u32 = .{ 0, 0 },
    world_pos: [2]f32 = .{ 0, 0 },

    pub fn update(self: *MouseButton, gctx: *zgpu.GraphicsContext) void {
        self.previousState = self.state;
        const action = gctx.window.getMouseButton(self.button);
        switch (action) {
            .release => {
                self.state = false;
            },
            .repeat, .press => {
                self.state = true;
            },
        }
        self.world_pos = getWorldPosition(gctx);
        self.grid_pos = getGridPosition(gctx);
        const content_scale = if (emscripten) .{ 1, 1 } else gctx.window.getContentScale();
        var pixel_pos = gctx.window.getCursorPos();
        pixel_pos[0] = @abs(pixel_pos[0]);
        pixel_pos[1] = @abs(pixel_pos[1]);
        self.pixel_pos = .{
            @as(u32, @intFromFloat(pixel_pos[0] * content_scale[0])),
            @as(u32, @intFromFloat(pixel_pos[1] * content_scale[1])),
        };
    }

    /// Returns true the frame the mouse button was pressed.
    pub fn pressed(self: MouseButton) bool {
        return self.state == true and self.state != self.previousState;
    }

    /// Returns true while the mouse button is pressed down.
    pub fn down(self: MouseButton) bool {
        return self.state == true;
    }

    /// Returns true the frame the mouse button was released.
    pub fn released(self: MouseButton) bool {
        return self.state == false and self.state != self.previousState;
    }

    /// Returns true while the mouse button is released.
    pub fn up(self: MouseButton) bool {
        return self.state == false;
    }
};

// Return world position of current cursor pos
pub fn getWorldPosition(gctx: *zgpu.GraphicsContext) [2]f32 {
    const viewport_size = Camera.getViewportPixelSize(gctx);
    const width = @as(f32, @floatFromInt(gctx.swapchain_descriptor.width));
    const xOffset = width - viewport_size[0];
    const cursor_pos = gctx.window.getCursorPos();
    const content_scale = if (emscripten) .{ 1, 1 } else gctx.window.getContentScale();
    const vp_cursor_pos = [2]f32{
        @as(f32, @floatCast(cursor_pos[0])) * content_scale[0] - xOffset,
        @as(f32, @floatCast(cursor_pos[1])) * content_scale[1],
    };

    const rx = (vp_cursor_pos[0] * 2) / viewport_size[0] - 1;
    const ry = 1 - (vp_cursor_pos[1] * 2) / viewport_size[1];
    const vec = zm.f32x4(rx, ry, 1, 1);
    const inv = zm.inverse(Camera.getObjectToClipMat(gctx));
    const world_pos = zm.mul(vec, inv);

    return .{
        world_pos[0] / world_pos[3],
        world_pos[1] / world_pos[3],
    };
}

pub fn getGridPosition(gctx: *zgpu.GraphicsContext) [2]i32 {
    const world_pos = getWorldPosition(gctx);
    const full_grid_pos = Camera.getGridFromWorld(gctx, world_pos);
    return .{ full_grid_pos[0], full_grid_pos[1] };
}

pub fn onGrid(gctx: *zgpu.GraphicsContext) bool {
    const grid_pos = getGridPosition(gctx);
    const x = grid_pos[0];
    const y = grid_pos[1];
    return x > Camera.MIN_X and x < Camera.MAX_X and y > Camera.MIN_Y and y < Camera.MAX_Y;
}
