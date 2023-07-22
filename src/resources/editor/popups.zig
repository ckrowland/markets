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
const ConsumerHover = @import("consumer_hover.zig");

pub const HoverBox = struct {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
};
const MIN_X = -70;
const MIN_Y = -70;
pub const CLOSED_SIZE = HoverBox {
    .min_x = MIN_X,
    .max_x = -MIN_X,
    .min_y = MIN_Y,
    .max_y = -MIN_Y,
};
pub const OPEN_SIZE = HoverBox {
    .min_x = MIN_X,
    .max_x = 600,
    .min_y = -390,
    .max_y = -MIN_Y,
};

const HoverSquareID = struct {
    hs_id: u32 = undefined,
    gui_id: u32,
};
pub const Popup = struct {
    id: HoverSquareID,
    // I'm done with calculating it all on the fly, just store center closed and open
    // Then push to github
    grid_agent_center: [2]i32,
    grid_gui_center: [2]i32 = undefined,
    grid_corners_closed: [4]i32 = undefined,
    grid_corners_open: [4]i32 = undefined,
    left_open: bool = false,
    open: bool = false,
    type_popup: enum { consumers, producer },
    parameters: union {
        consumer: struct {
            demand_rate: u32,
            moving_rate: f32,
        },
        producer: struct {
            production_rate: u32,
            max_inventory: u32,
        },
    },
};
pub const HoverSquare = struct {
    id: HoverSquareID,
    grid_corners: [4]i32,
};

const Self = @This();
hover_square_len: u32,
popups: std.ArrayList(Popup),
x_axis: std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),
y_axis: std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .hover_square_len = 0,
        .popups = std.ArrayList(Popup).init(allocator),
        .x_axis = std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)).init(allocator),
        .y_axis = std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)).init(allocator),
    };
}

pub fn deinit(self: *Self) void {
    self.popups.deinit();
    var x_arrays = self.x_axis.valueIterator();
    while (x_arrays.next()) |set| {
        set.deinit();
    }
    self.x_axis.deinit();
    
    var y_arrays = self.y_axis.valueIterator();
    while (y_arrays.next()) |set| {
        set.deinit();
    }
    self.y_axis.deinit();
}

pub fn clear(self: *Self) void {
    self.hover_square_len = 0;
    self.popups.clearAndFree();

    var x_arrays = self.x_axis.valueIterator();
    while (x_arrays.next()) |set| {
        set.clearAndFree();
    }
    self.x_axis.clearAndFree();
    
    var y_arrays = self.y_axis.valueIterator();
    while (y_arrays.next()) |set| {
        set.clearAndFree();
    }
    self.y_axis.clearAndFree();
}

pub fn appendPopup(self: *Self, popup: Popup) !void {
    var copy = popup;
    copy.id.hs_id = self.hover_square_len;

    const closed_edges = getGridEdges(popup.grid_agent_center, CLOSED_SIZE);
    copy.grid_corners_closed = closed_edges;
    
    var open_edges = getGridEdges(popup.grid_agent_center, OPEN_SIZE);
    var center = popup.grid_agent_center;
    if (open_edges[1] >= Camera.MAX_X) {
        const dist = open_edges[1] - open_edges[0];
        open_edges[1] = open_edges[1] - dist + CLOSED_SIZE.max_x;
        open_edges[0] = open_edges[0] - dist + CLOSED_SIZE.max_x;
        center = .{
            open_edges[0] + CLOSED_SIZE.max_x,
            open_edges[3] + CLOSED_SIZE.min_y,
        };
    }
    copy.grid_corners_open = open_edges;
    copy.grid_gui_center = center;
    
    try self.popups.append(copy);
}

pub fn appendSquare(self: *Self, allocator: std.mem.Allocator, square: HoverSquare) !void {
    var copy = square;
    copy.id.hs_id = self.hover_square_len;
    
    
    try self.appendRange(allocator, copy);
    self.hover_square_len += 1;
}

pub fn getGridEdges(grid_cursor: [2]i32, grid_hover_box: HoverBox) [4]i32 {
    return .{
        grid_cursor[0] + grid_hover_box.min_x,
        grid_cursor[0] + grid_hover_box.max_x,
        grid_cursor[1] + grid_hover_box.min_y,
        grid_cursor[1] + grid_hover_box.max_y,
    };
}

fn inBetween(bottom: i32, value: i32, top: i32) bool {
    return bottom < value and value < top;
}

fn resetCoords(
    axis: *std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),
    closed_grid: *const [2]i32,
    open_grid: *const [2]i32,
    hsid: HoverSquareID,
) void {
    var it = axis.iterator();
    while (it.next()) |kv| {
        const coord = kv.key_ptr.*;
        const pixel_has_gui = kv.value_ptr.contains(hsid);
        const in_open_grid = inBetween(open_grid[0], coord, open_grid[1]);
        const not_in_closed_grid = !inBetween(closed_grid[0], coord, closed_grid[1]);
        if (pixel_has_gui and in_open_grid and not_in_closed_grid) {
            _ = kv.value_ptr.remove(hsid);
            if (kv.value_ptr.count() == 0) {
                kv.value_ptr.deinit();
                _ = axis.remove(coord);
            }
        }
    }
}

fn addCoords(
    axis: *std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),
    allocator: std.mem.Allocator,
    range: *const [2]i32,
    hsid: HoverSquareID,
) !void {
    var coord = range[0];
    while (coord <= range[1]) {
        var set = axis.getPtr(coord);
        if (set) |s| {
            if (!s.contains(hsid)) {
                try s.put(hsid, {});
            }
        } else {
            var new_set = std.AutoHashMap(HoverSquareID, void).init(allocator);
            try new_set.put(hsid, {});
            try axis.put(coord, new_set);
        }
        coord += 1;
    }
}

fn resetRange(self: *Self, edges: [4]i32, gui_id: u32) void {
    resetCoords(&self.x_axis, edges[0..2], gui_id);
    resetCoords(&self.y_axis, edges[2..4], gui_id);
}
    
fn appendRange(self: *Self, allocator: std.mem.Allocator, hs: HoverSquare) !void {
    try addCoords(&self.x_axis, allocator, hs.grid_corners[0..2], hs.id);
    try addCoords(&self.y_axis, allocator, hs.grid_corners[2..4], hs.id);
}


fn printAxis(axis: *std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void))) void {
    var it = axis.iterator();
    while (it.next()) |kv| {
        std.debug.print("{any} -> ", .{kv.key_ptr.*});
        var set_it = kv.value_ptr.keyIterator();
        while (set_it.next()) |k| {
            std.debug.print("{any} ", .{k.*});
        }
        std.debug.print("\n", .{});
    }
        std.debug.print("\n", .{});
}


pub fn anyOpen(self: *Self) bool {
    var b = false;
    for (self.popups.items) |w| {
        b = b or w.open;
    }
    return b;
}

fn getPopupSizePixels(gctx: *zgpu.GraphicsContext, pop_up: *Popup) [2]f32 {
    const pixel_center = Camera.getPixelPosition(gctx, pop_up.grid_gui_center);
    const edges = pop_up.grid_corners_open;
    const top_right = Camera.getPixelPosition(gctx, .{ edges[1], edges[3] });
    const bottom_left = Camera.getPixelPosition(gctx, .{ edges[0], edges[2] });
    return .{
        top_right[0] - pixel_center[0],
        bottom_left[1] - pixel_center[1],
    };
}

fn isPopupOpen(self: *Self, gui_id: u32) bool {
    for (self.popups.items) |p| {
        if (p.id.gui_id == gui_id) {
            return p.open;
        }
    }
    return false;
}

fn getPopupIndex(self: *Self, grid_pos: [2]i32) !usize {
    var x_set = self.x_axis.get(grid_pos[0]);
    var y_set = self.y_axis.get(grid_pos[1]);
    if (x_set == null or y_set == null) {
        return error.PopupNotFound;
    }
    
    const gui_id: ?u32 = blk: {
        var last_gui_id: ?u32 = null;
        var it = x_set.?.keyIterator();
        while (it.next()) |key| {
            if (y_set.?.contains(key.*)) {
                last_gui_id = key.gui_id;
                if (isPopupOpen(self, key.gui_id)) {
                    break :blk key.gui_id;
                }
            }
        }

        // If neither gui is open, just open the last one we saw
        if (last_gui_id) |gid| {
            break :blk gid;
        }
        break :blk null;
    };
    
    if (gui_id == null) {
        return error.PopupNotFound;
    }
    if (gui_id.? >= self.popups.items.len) {
        return error.PopupNotFound;
    }
    return @intCast(usize, gui_id.?);
}

pub const popupArgs = struct {
    consumers: Wgpu.ObjectBuffer,
    consumer_hover: Wgpu.ObjectBuffer,
    mouse: Mouse.MouseButton,
    producers: Wgpu.ObjectBuffer,
    stats: Wgpu.ObjectBuffer,
    allocator: std.mem.Allocator,
};
pub fn display(self: *Self, gctx: *zgpu.GraphicsContext, args: popupArgs) !void {
    ConsumerHover.clearHover(gctx, .{
        .consumer_hover = args.consumer_hover,
        .stats = args.stats,
    });

    const popup_idx = getPopupIndex(self, args.mouse.grid_pos) catch {
        closeAllOpenPopups(self);
        return;
    };
    const popup = &self.popups.items[popup_idx];
    if (!popup.open) {
        try self.openPopup(args.allocator, popup);
    }
    ConsumerHover.highlightConsumers(gctx, popup_idx, .{
        .consumer_hover = args.consumer_hover,
        .stats = args.stats,
    });
    setupPopupWindow(gctx, popup);
    switch (popup.type_popup) {
        .consumers => consumerGui(gctx, popup_idx, popup, args),
        .producer => producerGui(gctx, popup_idx, popup, args),
    }
}

fn openPopup(self: *Self, allocator: std.mem.Allocator, popup: *Popup) !void {
    popup.open = true;
    try self.appendRange(allocator, .{
        .id = popup.id,
        .grid_corners = popup.grid_corners_open,
    });
}

fn closeAllOpenPopups(self: *Self) void {
    for (self.popups.items, 0..) |p, i| {
        if (p.open) {
            self.closePopup(&self.popups.items[i]);
        }
    }
}

fn closePopup(self: *Self, popup: *Popup) void {
    popup.open = false;
    resetCoords(
        &self.x_axis,
        popup.grid_corners_closed[0..2],
        popup.grid_corners_open[0..2],
        popup.id,
    );
    resetCoords(
        &self.y_axis,
        popup.grid_corners_closed[2..4],
        popup.grid_corners_open[2..4],
        popup.id,
    );
}

fn setupPopupWindow(gctx: *zgpu.GraphicsContext, pop_up: *Popup) void {
    const pixel_center = Camera.getPixelPosition(gctx, pop_up.grid_gui_center);
    const popup_size = getPopupSizePixels(gctx, pop_up);
    zgui.setNextWindowPos(.{
        .x = pixel_center[0],
        .y = pixel_center[1],
    });
    zgui.setNextWindowSize(.{
        .w = popup_size[0],
        .h = popup_size[1],
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
            .grouping_id = pop_up.id.gui_id,
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
            .grid_pos = pop_up.grid_agent_center,
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
            .grid_pos = pop_up.grid_agent_center,
        });
    }
}
