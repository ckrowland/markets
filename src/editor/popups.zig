const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const zmath = @import("zmath");
const Wgpu = @import("wgpu.zig");
const Windows = @import("windows.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Mouse = @import("mouse.zig");
const Camera = @import("camera.zig");
const ConsumerHover = @import("consumer_hover.zig");
const Callbacks = @import("callbacks.zig");

pub const WINDOW_SIZE_PIXELS: [2]u32 = .{ 400, 185 };
const HOVER_SIZE_GRID = 40;

const HoverSquareID = struct {
    hs_id: u32,
    gui_id: u32,
};
pub const HoverSquare = struct {
    id: HoverSquareID,
    corners_grid: [4]i32,
};
pub const Popup = struct {
    id: HoverSquareID = undefined,
    grid_center: [2]i32,
    pixel_center: [2]f32 = undefined,
    open: bool = false,
    pivot: bool = false,
    open_grid: [4]i32 = undefined,
    closed_grid: [4]i32 = undefined,
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

const Self = @This();
hover_square_len: u32,
popups: std.ArrayList(Popup),
x_axis: std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),
y_axis: std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .hover_square_len = 0,
        .popups = std.ArrayList(Popup).init(allocator),
        .x_axis = std.AutoHashMap(
            i32,
            std.AutoHashMap(HoverSquareID, void),
        ).init(allocator),
        .y_axis = std.AutoHashMap(
            i32,
            std.AutoHashMap(HoverSquareID, void),
        ).init(allocator),
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

pub fn appendPopup(self: *Self, popup: Popup) void {
    var copy = popup;
    copy.id.gui_id = @as(u32, @intCast(self.popups.items.len));
    copy.id.hs_id = self.hover_square_len;
    self.popups.append(copy) catch unreachable;
}

pub fn appendSquare(
    self: *Self,
    allocator: std.mem.Allocator,
    grid_pos: [2]i32,
) void {
    const gui_id = @as(u32, @intCast(self.popups.items.len));
    const square = HoverSquare{
        .id = .{
            .hs_id = self.hover_square_len,
            .gui_id = gui_id,
        },
        .corners_grid = .{
            grid_pos[0] - HOVER_SIZE_GRID,
            grid_pos[0] + HOVER_SIZE_GRID,
            grid_pos[1] - HOVER_SIZE_GRID,
            grid_pos[1] + HOVER_SIZE_GRID,
        },
    };
    self.appendRange(allocator, square);
    self.hover_square_len += 1;
}

fn inBetween(bottom: i32, value: i32, top: i32) bool {
    return bottom < value and value < top;
}

fn resetCoords(
    axis: *std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),
    closed_grid: *[2]i32,
    open_grid: *[2]i32,
    hsid: HoverSquareID,
) void {
    var it = axis.iterator();
    while (it.next()) |kv| {
        const coord = kv.key_ptr.*;
        const pixel_has_gui = kv.value_ptr.contains(hsid);
        const in_open_grid = inBetween(open_grid[0], coord, open_grid[1]);
        const not_in_closed_grid = !inBetween(
            closed_grid[0],
            coord,
            closed_grid[1],
        );
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
        const set = axis.getPtr(coord);
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

fn appendRange(self: *Self, allocator: std.mem.Allocator, hs: HoverSquare) void {
    addCoords(&self.x_axis, allocator, hs.corners_grid[0..2], hs.id) catch unreachable;
    addCoords(&self.y_axis, allocator, hs.corners_grid[2..4], hs.id) catch unreachable;
}

pub fn anyOpen(self: *Self) bool {
    var b = false;
    for (self.popups.items) |w| {
        b = b or w.open;
    }
    return b;
}

fn isPopupOpen(self: *Self, gui_id: u32) bool {
    for (self.popups.items) |p| {
        if (p.id.gui_id == gui_id) {
            return p.open;
        }
    }
    return false;
}

pub fn doesAgentExist(self: *Self, grid_pos: [2]i32) bool {
    var x_set = self.x_axis.get(grid_pos[0]);
    var y_set = self.y_axis.get(grid_pos[1]);
    if (x_set == null or y_set == null) {
        return false;
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
        return false;
    }
    return true;
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
    return @as(usize, @intCast(gui_id.?));
}

pub const popupArgs = struct {
    consumers: *Wgpu.ObjectBuffer(Consumer),
    consumer_hovers: *Wgpu.ObjectBuffer(ConsumerHover),
    mouse: Mouse.MouseButton,
    producers: Wgpu.ObjectBuffer(Producer),
    stats: Wgpu.ObjectBuffer(u32),
    allocator: std.mem.Allocator,
    content_scale: f32,
};
pub fn display(self: *Self, gctx: *zgpu.GraphicsContext, args: popupArgs) void {
    ConsumerHover.clearHover(gctx, args.consumer_hovers);

    const popup_idx = getPopupIndex(self, args.mouse.grid_pos) catch {
        for (self.popups.items, 0..) |p, i| {
            if (p.open) {
                self.closePopup(&self.popups.items[i]);
            }
        }
        return;
    };

    ConsumerHover.highlightConsumers(
        gctx,
        popup_idx,
        args.consumer_hovers,
    );

    const popup = &self.popups.items[popup_idx];
    if (!popup.open) {
        popup.open = true;

        var center = popup.grid_center;
        const pixel_center = Camera.getPixelPosition(gctx, center);
        const min_x_pixel = pixel_center[0] - HOVER_SIZE_GRID;
        const max_x_pixel = pixel_center[0] + (WINDOW_SIZE_PIXELS[0] * args.content_scale);
        const min_y_pixel = pixel_center[1] - HOVER_SIZE_GRID;
        const max_y_pixel = pixel_center[1] + (WINDOW_SIZE_PIXELS[1] * args.content_scale);

        var open_grid: [4]i32 = .{
            popup.grid_center[0] - HOVER_SIZE_GRID,
            Camera.getGridPosition(gctx, .{ max_x_pixel, min_y_pixel })[0],
            Camera.getGridPosition(gctx, .{ min_x_pixel, max_y_pixel })[1],
            popup.grid_center[1] + HOVER_SIZE_GRID,
        };

        if (open_grid[1] >= Camera.MAX_X) {
            const len_x = open_grid[1] - open_grid[0] - (2 * HOVER_SIZE_GRID);
            open_grid[1] = open_grid[0] + (2 * HOVER_SIZE_GRID);
            open_grid[0] -= len_x;
            center = .{
                open_grid[1] - HOVER_SIZE_GRID,
                open_grid[3] - HOVER_SIZE_GRID,
            };
            popup.pivot = true;
        }

        popup.pixel_center = pixel_center;
        popup.open_grid = open_grid;

        self.appendRange(args.allocator, .{
            .id = popup.id,
            .corners_grid = popup.open_grid,
        });
    }

    if (popup.pivot) {
        zgui.setNextWindowPos(.{
            .x = popup.pixel_center[0],
            .y = popup.pixel_center[1],
            .pivot_x = 1.0,
        });
    } else {
        zgui.setNextWindowPos(.{
            .x = popup.pixel_center[0],
            .y = popup.pixel_center[1],
        });
    }

    zgui.setNextWindowSize(.{
        .w = WINDOW_SIZE_PIXELS[0] * args.content_scale,
        .h = WINDOW_SIZE_PIXELS[1] * args.content_scale,
    });

    switch (popup.type_popup) {
        .consumers => consumerGui(gctx, popup_idx, popup, args),
        .producer => producerGui(gctx, popup_idx, popup, args),
    }
}

fn closePopup(self: *Self, popup: *Popup) void {
    popup.open = false;
    resetCoords(
        &self.x_axis,
        popup.closed_grid[0..2],
        popup.open_grid[0..2],
        popup.id,
    );
    resetCoords(
        &self.y_axis,
        popup.closed_grid[2..4],
        popup.open_grid[2..4],
        popup.id,
    );
}

fn consumerGui(gctx: *zgpu.GraphicsContext, idx: usize, popup: *Popup, args: popupArgs) void {
    if (zgui.begin("Test", Windows.window_flags)) {
        zgui.pushIntId(@as(i32, @intCast(idx)) + 3);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        demandRateButton(gctx, popup, args);
        movingRateSlider(gctx, popup, args);
        zgui.popId();
    }
    zgui.end();
}

fn demandRateButton(gctx: *zgpu.GraphicsContext, popup: *Popup, args: popupArgs) void {
    zgui.text("Demand Rate", .{});
    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("How much consumers demand from producers on a single trip.");
        zgui.endTooltip();
    }

    const demand_rate_ptr = &popup.parameters.consumer.demand_rate;
    if (zgui.sliderScalar(
        "##dr",
        u32,
        .{ .v = demand_rate_ptr, .min = 1, .max = 1000 },
    )) {
        for (args.consumers.list.items, 0..) |c, i| {
            if (popup.id.gui_id == c.grouping_id) {
                gctx.queue.writeBuffer(
                    gctx.lookupResource(args.consumers.buf).?,
                    i * @sizeOf(Consumer) + @offsetOf(Consumer, "demand_rate"),
                    u32,
                    &.{demand_rate_ptr.*},
                );
            }
        }
    }
}

fn movingRateSlider(gctx: *zgpu.GraphicsContext, popup: *Popup, args: popupArgs) void {
    zgui.text("Moving Rate", .{});
    const moving_rate_ptr = &popup.parameters.consumer.moving_rate;
    if (zgui.sliderScalar("##mr", f32, .{ .v = moving_rate_ptr, .min = 1.0, .max = 20 })) {
        for (args.consumers.list.items, 0..) |c, i| {
            if (popup.id.gui_id == c.grouping_id) {
                gctx.queue.writeBuffer(
                    gctx.lookupResource(args.consumers.buf).?,
                    i * @sizeOf(Consumer) + @offsetOf(Consumer, "moving_rate"),
                    f32,
                    &.{moving_rate_ptr.*},
                );
            }
        }
    }
}

fn producerGui(gctx: *zgpu.GraphicsContext, idx: usize, popup: *Popup, args: popupArgs) void {
    if (zgui.begin("Test", Windows.window_flags)) {
        zgui.pushIntId(@as(i32, @intCast(idx)) + 3);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        productionRateButton(gctx, popup, args);
        maxInventoryButton(gctx, popup, args);
        zgui.popId();
    }
    zgui.end();
}

fn productionRateButton(gctx: *zgpu.GraphicsContext, popup: *Popup, args: popupArgs) void {
    zgui.text("Production Rate", .{});
    const production_rate_ptr = &popup.parameters.producer.production_rate;
    if (zgui.sliderScalar("##pr", u32, .{ .v = production_rate_ptr, .min = 1, .max = 1000 })) {
        for (args.producers.list.items, 0..) |p, i| {
            if (std.mem.eql(i32, popup.grid_center[0..2], p.absolute_home[0..2])) {
                gctx.queue.writeBuffer(
                    gctx.lookupResource(args.producers.buf).?,
                    i * @sizeOf(Producer) + @offsetOf(Producer, "production_rate"),
                    u32,
                    &.{production_rate_ptr.*},
                );
            }
        }
    }
}

fn maxInventoryButton(gctx: *zgpu.GraphicsContext, popup: *Popup, args: popupArgs) void {
    zgui.text("Max Inventory", .{});
    const max_inventory_ptr = &popup.parameters.producer.max_inventory;
    if (zgui.sliderScalar("##mi", u32, .{ .v = max_inventory_ptr, .min = 10, .max = 10000 })) {
        for (args.producers.list.items, 0..) |p, i| {
            if (std.mem.eql(i32, popup.grid_center[0..2], p.absolute_home[0..2])) {
                gctx.queue.writeBuffer(
                    gctx.lookupResource(args.producers.buf).?,
                    i * @sizeOf(Producer) + @offsetOf(Producer, "max_inventory"),
                    u32,
                    &.{max_inventory_ptr.*},
                );
            }
        }
    }
}
