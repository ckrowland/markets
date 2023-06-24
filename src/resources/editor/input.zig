const std = @import("std");
const zglfw = @import("zglfw");
const zmath = @import("zmath");
const zgpu = @import("zgpu");
const main = @import("main.zig");
const Camera = @import("../../camera.zig");

pub const MouseButton = struct {
    name: [:0]const u8 = "Primary",
    button: zglfw.MouseButton = zglfw.MouseButton.left,
    state: bool = false,
    previousState: bool = false,

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
pub fn getWorldPosition(gctx: *zgpu.GraphicsContext) zmath.F32x4 {
    const width = @floatFromInt(f64, gctx.swapchain_descriptor.width);
    const height = @floatFromInt(f64, gctx.swapchain_descriptor.height);
    const xOffset = width - (width * Camera.VP_X_SIZE);
    const viewport_size = .{ width * Camera.VP_X_SIZE, height * Camera.VP_Y_SIZE };
    const cursor_pos = gctx.window.getCursorPos();
    const cursor_pos_in_vp = .{ cursor_pos[0] - (xOffset / 2), cursor_pos[1] };

    const rx = (cursor_pos_in_vp[0] * 4) / viewport_size[0] - 1;
    const ry = 1 - (cursor_pos_in_vp[1] * 4) / viewport_size[1];
    const x = @floatCast(f32, rx);
    const y = @floatCast(f32, ry);
    const vec = zmath.f32x4(x * -Camera.POS_Z, y * -Camera.POS_Z, 0, 1);
    const inv = zmath.inverse(Camera.getObjectToClipMat(gctx));
    return zmath.mul(vec, inv);
}

pub fn onGrid(gctx: *zgpu.GraphicsContext) bool {
    const world_pos = getWorldPosition(gctx);
    const grid_pos = Camera.getGridPosition(gctx, world_pos);
    const x = grid_pos[0];
    const y = grid_pos[1];
    return x > Camera.MIN_X and x < Camera.MAX_X and y > Camera.MIN_Y and y < Camera.MAX_Y;
}
