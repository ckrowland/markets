const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const zmath = @import("zmath");
const Wgpu = @import("../wgpu.zig");
const Windows = @import("../../windows.zig");
const Producer = @import("../producer.zig");
const Mouse = @import("mouse.zig");
const Camera = @import("../../camera.zig");

pub const INITIAL_HOVER_SIZE: [4]f32 = .{ -30, 30, -30, 30 };
pub const Popup = struct {
    window_cursor_pos: [2]f32,
    grid_cursor_pos: [2]f32,
    initial_hover_size: [4]f32 = INITIAL_HOVER_SIZE,
    grid_hover_size: [4]f32 = INITIAL_HOVER_SIZE,
    grid_popup_size: [4]f32 = .{ -30, 600, -600, 30 },
    open: bool = false,
};

const Self = @This();
positions: std.ArrayList(Popup),

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .positions = std.ArrayList(Popup).init(allocator),
    };
}

pub fn deinit(self: Self) void {
    self.positions.deinit();
}

pub fn addWindow(self: *Self, pop_up: Popup) void {
    self.positions.append(pop_up) catch unreachable;
}

pub fn anyOpen(self: *Self) bool {
    var b = false;
    for (self.positions.items) |w| {
        b = b or w.open;
    }
    return b;
}

const PixelCorners = struct {
    top_left: [2]f32,
    top_right: [2]f32,
    bottom_left: [2]f32,
};
fn getPixelCorners(gctx: *zgpu.GraphicsContext, pop_up: *Popup) PixelCorners {
    const min_x = pop_up.grid_cursor_pos[0] + pop_up.grid_hover_size[0];
    const max_x = pop_up.grid_cursor_pos[0] + pop_up.grid_hover_size[1];
    const min_y = pop_up.grid_cursor_pos[1] + pop_up.grid_hover_size[2];
    const max_y = pop_up.grid_cursor_pos[1] + pop_up.grid_hover_size[3];
    return .{
        .top_left = Camera.getPixelPosition(gctx, .{ min_x, min_y }),
        .top_right = Camera.getPixelPosition(gctx, .{ max_x, min_y }),
        .bottom_left = Camera.getPixelPosition(gctx, .{ min_x, max_y }),
    };
}

fn getPopupSize(gctx: *zgpu.GraphicsContext, pop_up: *Popup) [2]f32 {
    const corners = getPixelCorners(gctx, pop_up);
    return .{
        corners.top_right[0] - corners.top_left[0],
        corners.bottom_left[1] - corners.top_left[1],
    };
}

// Return true if mouse pos is hovering over agent or popup
fn mouseOverPopup(gctx: *zgpu.GraphicsContext, mouse: Mouse.MouseButton, pop_up: *Popup) bool {
    const pixel_corners = getPixelCorners(gctx, pop_up);

    const top_left = pixel_corners.top_left;
    const top_right = pixel_corners.top_right;
    const bottom_left = pixel_corners.bottom_left;

    const in_x = top_left[0] < mouse.cursor_pos[0] and mouse.cursor_pos[0] < top_right[0];
    const in_y = bottom_left[1] < mouse.cursor_pos[1] and mouse.cursor_pos[1] < top_left[1];
    return in_x and in_y;
}

pub const ProducerArgs = struct {
    const parameters = struct {
        production_rate: *u32,
        max_inventory: *u32,
    };
    params_ref: parameters,
    agents: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
    num_agents: u32,
};
pub fn display(
    self: Self,
    gctx: *zgpu.GraphicsContext,
    mouse: Mouse.MouseButton,
    comptime T: type,
    args: ProducerArgs,
) void {
    _ = T;
    for (self.positions.items, 0..) |pop_up, i| {
        if (pop_up.open) {
            self.positions.items[i].grid_hover_size = pop_up.grid_popup_size;
        }

        const popup_ref = &self.positions.items[i];
        if (mouseOverPopup(gctx, mouse, popup_ref)) {
            self.positions.items[i].open = true;
            setupPopupWindow(gctx, popup_ref);
            producerGui(gctx, i, args);
        } else {
            self.positions.items[i].open = false;
            self.positions.items[i].grid_hover_size = INITIAL_HOVER_SIZE;
        }
    }
}

fn setupPopupWindow(gctx: *zgpu.GraphicsContext, pop_up: *Popup) void {
    const coords = Camera.getPixelPosition(gctx, pop_up.grid_cursor_pos);
    zgui.setNextWindowPos(.{
        .x = coords[0],
        .y = coords[1],
    });
    zgui.setNextWindowSize(.{
        .w = getPopupSize(gctx, pop_up)[0],
        .h = getPopupSize(gctx, pop_up)[1],
    });
}
fn producerGui(gctx: *zgpu.GraphicsContext, idx: usize, args: ProducerArgs) void {
    if (zgui.begin("Test", Windows.window_flags)) {
        zgui.pushIntId(@intCast(i32, idx) + 3);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        productionRateButton(gctx, args);
        maxInventoryButton(gctx, args);
        zgui.popId();
    }
    zgui.end();
}

fn productionRateButton(gctx: *zgpu.GraphicsContext, args: ProducerArgs) void {
    zgui.text("Production Rate", .{});
    if (zgui.sliderScalar("##pr", u32, .{
        .v = args.params_ref.production_rate,
        .min = 1,
        .max = 1000,
    })) {
        Wgpu.setAll(gctx, Producer, .{
            .agents = args.agents,
            .stats = args.stats,
            .num_agents = args.num_agents,
            .parameter = .{
                .production_rate = args.params_ref.production_rate.*,
            },
        });
    }
}

fn maxInventoryButton(gctx: *zgpu.GraphicsContext, args: ProducerArgs) void {
    zgui.text("Max Inventory", .{});
    if (zgui.sliderScalar("##mi", u32, .{
        .v = args.params_ref.max_inventory,
        .min = 10,
        .max = 10000,
    })) {
        Wgpu.setAll(gctx, Producer, .{
            .agents = args.agents,
            .stats = args.stats,
            .num_agents = args.num_agents,
            .parameter = .{
                .max_inventory = args.params_ref.max_inventory.*,
            },
        });
    }
}
