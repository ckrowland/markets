const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const zmath = @import("zmath");
const Wgpu = @import("../wgpu.zig");
const Windows = @import("../../windows.zig");
const Consumer = @import("../consumer.zig");
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
    type_popup: enum { consumers, producer },
    parameters: union {
        consumer: struct {
            demand_rate: u32,
            moving_rate: f32,
            grouping_id: u32,
        },
        producer: struct {
            production_rate: u32,
            max_inventory: u32,
        },
    },
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

pub fn append(self: *Self, pop_up: Popup) void {
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

pub const popupArgs = struct {
    consumers: Wgpu.ObjectBuffer,
    mouse: Mouse.MouseButton,
    producers: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
};
pub fn display(self: Self, gctx: *zgpu.GraphicsContext, args: popupArgs) void {
    for (self.positions.items, 0..) |pop_up, i| {
        if (pop_up.open) {
            self.positions.items[i].grid_hover_size = pop_up.grid_popup_size;
        }

        const popup_ref = &self.positions.items[i];
        if (mouseOverPopup(gctx, args.mouse, popup_ref)) {
            self.positions.items[i].open = true;
            setupPopupWindow(gctx, popup_ref);
            switch (pop_up.type_popup) {
                .consumers => consumerGui(gctx, i, popup_ref, args),
                .producer => producerGui(gctx, i, popup_ref, args),
            }
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

fn consumerGui(gctx: *zgpu.GraphicsContext, idx: usize, pop_up: *Popup, args: popupArgs) void {
    if (zgui.begin("Test", Windows.window_flags)) {
        zgui.pushIntId(@intCast(i32, idx) + 3);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        demandRateButton(gctx, pop_up, args);
        movingRateSlider(gctx, pop_up, args);
        zgui.popId();
    }
    zgui.end();
}

fn demandRateButton(gctx: *zgpu.GraphicsContext, pop_up: *Popup, args: popupArgs) void {
    zgui.text("Demand Rate", .{});
    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("How much consumers demand from producers on a single trip.");
        zgui.endTooltip();
    }

    const demand_rate_ptr = &pop_up.parameters.consumer.demand_rate;
    if (zgui.sliderScalar("##dr", u32, .{ .v = demand_rate_ptr, .min = 1, .max = 1000 },)) {
        Wgpu.setAll(gctx, Consumer, .{
            .agents = args.consumers,
            .stats = args.stats,
            .parameter = .{
                .demand_rate = demand_rate_ptr.*,
            },
        });
    }
}

fn movingRateSlider(gctx: *zgpu.GraphicsContext, pop_up: *Popup, args: popupArgs) void {
    zgui.text("Moving Rate", .{});
    const moving_rate_ptr = &pop_up.parameters.consumer.moving_rate;
    if (zgui.sliderScalar("##mr", f32, .{ .v = moving_rate_ptr, .min = 1.0, .max = 20 })) {
        Wgpu.setGroup(gctx, Consumer, .{
            .setArgs = .{
                .agents = args.consumers,
                .stats = args.stats,
                .parameter = .{
                    .moving_rate = moving_rate_ptr.*,
                },
            },
            .grouping_id = pop_up.parameters.consumer.grouping_id,
        });
    }
}

fn producerGui(gctx: *zgpu.GraphicsContext, idx: usize, pop_up: *Popup, args: popupArgs) void {
    if (zgui.begin("Test", Windows.window_flags)) {
        zgui.pushIntId(@intCast(i32, idx) + 3);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        productionRateButton(gctx, pop_up, args);
        maxInventoryButton(gctx, pop_up, args);
        zgui.popId();
    }
    zgui.end();
}

fn productionRateButton(gctx: *zgpu.GraphicsContext, pop_up: *Popup, args: popupArgs) void {
    zgui.text("Production Rate", .{});
    const production_rate_ptr = &pop_up.parameters.producer.production_rate;
    if (zgui.sliderScalar("##pr", u32, .{.v = production_rate_ptr, .min = 1, .max = 1000})) {
        Wgpu.setAgent(gctx, Producer, .{
            .setArgs = .{
                .agents = args.producers,
                .stats = args.stats,
                .parameter = .{
                    .production_rate = production_rate_ptr.*,
                },
            },
            .grid_pos = pop_up.grid_cursor_pos,
        });
    }
}

fn maxInventoryButton(gctx: *zgpu.GraphicsContext, pop_up: *Popup, args: popupArgs) void {
    zgui.text("Max Inventory", .{});
    const max_inventory_ptr = &pop_up.parameters.producer.max_inventory;
    if (zgui.sliderScalar("##mi", u32, .{.v = max_inventory_ptr, .min = 10, .max = 10000})) {
        Wgpu.setAgent(gctx, Producer, .{
            .setArgs = .{
                .agents = args.producers,
                .stats = args.stats,
                .parameter = .{
                    .max_inventory = max_inventory_ptr.*,
                },
            },
            .grid_pos = pop_up.grid_cursor_pos,
        });
    }
}
