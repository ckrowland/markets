const std = @import("std");
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const Camera = @import("camera");
const Wgpu = @import("wgpu");
const Consumer = @import("consumer");
const Producer = @import("producer");
const Mouse = @import("mouse.zig");
const ConsumerHover = @import("consumer_hover.zig");
const Callbacks = @import("callbacks.zig");

pub const WINDOW_SIZE_PIXELS: [2]u32 = .{ 400, 185 };
const HOVER_SIZE_GRID = 30;

pub const AsyncAxisWrite = struct {
    start: i32 = 0,
    val: i32 = 0,
    end: i32 = 0,
    done: bool = true,
    max_num_writes: u32 = 70,
};
pub const HoverSquareID = struct {
    hs_id: ?u32 = null,
    gui_id: u32 = undefined,
    popup_type: PopupType = undefined,
};
pub const HoverSquare = struct {
    id: HoverSquareID = .{},
    corners_grid: [4]i32 = .{ 0, 0, 0, 0 },
    async_write: [2]AsyncAxisWrite = .{ .{}, .{} },
};
pub const PopupType = enum { consumers, producer };
pub const Popup = struct {
    hs: HoverSquare = .{},
    grid_center: [4]i32,
    pixel_center: [2]f32 = undefined,
    open: bool = false,
    pivot: bool = false,
    open_grid: [4]i32 = undefined,
    closed_grid: [4]i32 = undefined,
    type_popup: PopupType,
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
consumers_popups: std.ArrayList(Popup),
producer_popups: std.ArrayList(Popup),
x_axis: std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),
y_axis: std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .consumers_popups = std.ArrayList(Popup).init(allocator),
        .producer_popups = std.ArrayList(Popup).init(allocator),
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
    self.consumers_popups.deinit();
    self.producer_popups.deinit();
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
    self.consumers_popups.clearAndFree();
    self.producer_popups.clearAndFree();

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
    switch (popup.type_popup) {
        .consumers => {
            copy.hs.id.gui_id = @as(u32, @intCast(self.consumers_popups.items.len));
            copy.hs.id.popup_type = .consumers;
            self.consumers_popups.append(copy) catch unreachable;
        },
        .producer => {
            copy.hs.id.gui_id = @as(u32, @intCast(self.producer_popups.items.len));
            copy.hs.id.popup_type = .producer;
            self.producer_popups.append(copy) catch unreachable;
        },
    }
}

pub fn appendSquare(
    self: *Self,
    allocator: std.mem.Allocator,
    grid_pos: [4]i32,
    popup_type: PopupType,
) void {
    const gui_id = switch (popup_type) {
        .consumers => @as(u32, @intCast(self.consumers_popups.items.len)),
        .producer => @as(u32, @intCast(self.producer_popups.items.len)),
    };

    var square = HoverSquare{
        .id = .{
            .hs_id = self.x_axis.count() + self.y_axis.count(),
            .gui_id = gui_id,
            .popup_type = popup_type,
        },
        .corners_grid = .{
            grid_pos[0] - HOVER_SIZE_GRID,
            grid_pos[0] + HOVER_SIZE_GRID,
            grid_pos[1] - HOVER_SIZE_GRID,
            grid_pos[1] + HOVER_SIZE_GRID,
        },
    };
    square.async_write = .{
        .{
            .start = square.corners_grid[0],
            .val = square.corners_grid[0],
            .end = square.corners_grid[1],
            .done = false,
        },
        .{
            .start = square.corners_grid[2],
            .val = square.corners_grid[2],
            .end = square.corners_grid[3],
            .done = false,
        },
    };
    self.appendRange(allocator, &square);
}

fn inBetween(bottom: i32, value: i32, top: i32) bool {
    return bottom < value and value < top;
}

fn addCoords(
    axis: *std.AutoHashMap(i32, std.AutoHashMap(HoverSquareID, void)),
    allocator: std.mem.Allocator,
    write: *AsyncAxisWrite,
    hs: HoverSquare,
) !void {
    var num_writes: u32 = 0;
    while (write.val <= write.end) {
        if (num_writes >= write.max_num_writes) return;

        const set = axis.getPtr(write.val);
        if (set) |s| {
            if (!s.contains(hs.id)) {
                try s.put(hs.id, {});
            }
        } else {
            var new_set = std.AutoHashMap(HoverSquareID, void).init(allocator);
            try new_set.put(hs.id, {});
            try axis.put(write.val, new_set);
        }

        write.val += 1;
        num_writes += 1;
    }
    write.done = true;
}

fn appendRange(self: *Self, allocator: std.mem.Allocator, hs: *HoverSquare) void {
    addCoords(&self.x_axis, allocator, &hs.async_write[0], hs.*) catch unreachable;
    addCoords(&self.y_axis, allocator, &hs.async_write[1], hs.*) catch unreachable;
}

pub fn anyOpen(self: *Self) bool {
    for (self.consumers_popups.items) |w| {
        if (w.open) return true;
    }
    for (self.producer_popups.items) |w| {
        if (w.open) return true;
    }
    return false;
}

fn isPopupOpen(self: *Self, hsid: HoverSquareID) bool {
    switch (hsid.popup_type) {
        .consumers => {
            for (self.consumers_popups.items) |p| {
                if (p.id.gui_id == hsid.gui_id) {
                    return p.open;
                }
            }
        },
        .producer => {
            for (self.producer_popups.items) |p| {
                if (p.id.gui_id == hsid.gui_id) {
                    return p.open;
                }
            }
        },
    }
    return false;
}

pub fn doesAgentExist(self: *Self, grid_pos: [4]i32) bool {
    _ = self.getHoverSquare(grid_pos) catch {
        return false;
    };
    return true;
}

fn getHoverSquare(self: *Self, grid_pos: [4]i32) !HoverSquareID {
    var x_set = self.x_axis.get(grid_pos[0]);
    var y_set = self.y_axis.get(grid_pos[1]);
    if (x_set == null or y_set == null) {
        return error.HSIDNotFound;
    }
    var it = x_set.?.keyIterator();
    while (it.next()) |key| {
        if (y_set.?.contains(key.*)) {
            return key.*;
        }
    }

    return error.HSIDNotFound;
}

fn getPopup(self: *Self, grid_pos: [4]i32) !*Popup {
    const hsid = try self.getHoverSquare(grid_pos);
    const gui_id = hsid.gui_id;
    switch (hsid.popup_type) {
        .consumers => {
            if (gui_id >= self.consumers_popups.items.len) {
                return error.GuiIDTooBig;
            }
            return &self.consumers_popups.items[gui_id];
        },
        .producer => {
            if (gui_id >= self.producer_popups.items.len) {
                return error.GuiIDTooBig;
            }
            return &self.producer_popups.items[gui_id];
        },
    }
}

fn closePopup(self: *Self, p: *Popup) void {
    if (p.open == false) return;
    p.open = false;

    var it = self.x_axis.iterator();
    while (it.next()) |kv| {
        const coord = kv.key_ptr.*;
        const pixel_has_gui = kv.value_ptr.contains(p.hs.id);
        const in_open_grid = p.open_grid[0] < coord and coord < p.open_grid[1];
        const not_in_closed_grid = coord < p.closed_grid[0] or p.closed_grid[1] < coord;
        if (pixel_has_gui and in_open_grid and not_in_closed_grid) {
            _ = kv.value_ptr.remove(p.hs.id);
            if (kv.value_ptr.count() == 0) {
                kv.value_ptr.deinit();
                _ = self.x_axis.remove(coord);
            }
        }
    }

    it = self.y_axis.iterator();
    while (it.next()) |kv| {
        const coord = kv.key_ptr.*;
        const pixel_has_gui = kv.value_ptr.contains(p.hs.id);
        const in_open_grid = p.open_grid[2] < coord and coord < p.open_grid[3];
        const not_in_closed_grid = coord < p.closed_grid[2] or p.closed_grid[3] < coord;
        if (pixel_has_gui and in_open_grid and not_in_closed_grid) {
            _ = kv.value_ptr.remove(p.hs.id);
            if (kv.value_ptr.count() == 0) {
                kv.value_ptr.deinit();
                _ = self.y_axis.remove(coord);
            }
        }
    }
}

pub const popupArgs = struct {
    consumers: *Wgpu.ObjectBuffer(Consumer),
    consumer_params: zgpu.BufferHandle,
    consumer_hovers: *Wgpu.ObjectBuffer(ConsumerHover),
    consumer_hover_colors: zgpu.BufferHandle,
    mouse: Mouse.MouseButton,
    producers: Wgpu.ObjectBuffer(Producer),
    allocator: std.mem.Allocator,
    content_scale: f32,
};

fn closeAllPopups(self: *Self, gctx: *zgpu.GraphicsContext, args: popupArgs) void {
    const resource = gctx.lookupResource(args.consumer_hover_colors).?;
    const black = [4]f32{ 0, 0, 0, 0 };
    for (self.consumers_popups.items[0..]) |*p| {
        const offset = p.hs.id.gui_id * 4 * @sizeOf(f32);
        gctx.queue.writeBuffer(resource, offset, [4]f32, &.{black});
        self.closePopup(p);
    }
    for (self.producer_popups.items[0..]) |*p| {
        self.closePopup(p);
    }
}

fn checkAsyncWrites(self: *Self, alloc: std.mem.Allocator, p: *Popup) void {
    const writes = &p.hs.async_write;
    if (!writes[0].done) {
        addCoords(&self.x_axis, alloc, &writes[0], p.hs) catch unreachable;
    }
    if (!writes[1].done) {
        addCoords(&self.y_axis, alloc, &writes[1], p.hs) catch unreachable;
    }
}

pub fn display(self: *Self, gctx: *zgpu.GraphicsContext, args: popupArgs) void {
    for (self.consumers_popups.items[0..]) |*p| {
        checkAsyncWrites(self, args.allocator, p);
    }

    for (self.producer_popups.items[0..]) |*p| {
        checkAsyncWrites(self, args.allocator, p);
    }

    var popup = getPopup(self, args.mouse.grid_pos) catch {
        closeAllPopups(self, gctx, args);
        return;
    };

    if (popup.type_popup == .consumers) {
        const resource = gctx.lookupResource(args.consumer_hover_colors).?;
        const blue = [4]f32{ 0, 0, 1, 0 };
        const offset = popup.hs.id.gui_id * 4 * @sizeOf(f32);
        gctx.queue.writeBuffer(resource, offset, [4]f32, &.{blue});
    }

    if (!popup.open) {
        popup.open = true;

        var center = popup.grid_center;
        const pixel_center = Camera.getPixelPosition(gctx, center[0..2].*);
        const min_x_pixel = pixel_center[0] - HOVER_SIZE_GRID;
        const max_x_pixel = pixel_center[0] + (WINDOW_SIZE_PIXELS[0] * args.content_scale);
        const min_y_pixel = pixel_center[1] - HOVER_SIZE_GRID;
        const max_y_pixel = pixel_center[1] + (WINDOW_SIZE_PIXELS[1] * args.content_scale);

        var open_grid: [4]i32 = .{
            popup.grid_center[0] - HOVER_SIZE_GRID,
            Camera.getGridPosition(gctx, .{ max_x_pixel, min_y_pixel, 0, 1 })[0],
            Camera.getGridPosition(gctx, .{ min_x_pixel, max_y_pixel, 0, 1 })[1],
            popup.grid_center[1] + HOVER_SIZE_GRID,
        };

        if (open_grid[1] >= Camera.MAX_X) {
            const len_x = open_grid[1] - open_grid[0] - (2 * HOVER_SIZE_GRID);
            open_grid[1] = open_grid[0] + (2 * HOVER_SIZE_GRID);
            open_grid[0] -= len_x;
            center = .{
                open_grid[1] - HOVER_SIZE_GRID,
                open_grid[3] - HOVER_SIZE_GRID,
                0,
                1,
            };
            popup.pivot = true;
        }

        popup.pixel_center = pixel_center;
        popup.open_grid = open_grid;
        popup.hs.corners_grid = popup.open_grid;

        popup.hs.async_write = .{
            .{
                .start = popup.open_grid[0],
                .val = popup.open_grid[0],
                .end = popup.open_grid[1],
                .done = false,
            },
            .{
                .start = popup.open_grid[2],
                .val = popup.open_grid[2],
                .end = popup.open_grid[3],
                .done = false,
            },
        };

        self.appendRange(args.allocator, &popup.hs);
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

    const flags = zgui.WindowFlags.no_decoration;
    if (zgui.begin("Test", .{ .flags = flags })) {
        switch (popup.type_popup) {
            .consumers => {
                zgui.pushIntId(@as(i32, @intCast(popup.hs.id.gui_id)) + 3);
                zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
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
                    const offset = popup.hs.id.gui_id * @sizeOf(Consumer.Params) +
                        @offsetOf(Consumer.Params, "demand_rate");
                    gctx.queue.writeBuffer(
                        gctx.lookupResource(args.consumer_params).?,
                        offset,
                        u32,
                        &.{demand_rate_ptr.*},
                    );
                }

                zgui.text("Moving Rate", .{});
                const moving_rate_ptr = &popup.parameters.consumer.moving_rate;
                if (zgui.sliderScalar("##mr", f32, .{ .v = moving_rate_ptr, .min = 1.0, .max = 20 })) {
                    const offset = popup.hs.id.gui_id * @sizeOf(Consumer.Params) +
                        @offsetOf(Consumer.Params, "moving_rate");
                    gctx.queue.writeBuffer(
                        gctx.lookupResource(args.consumer_params).?,
                        offset,
                        f32,
                        &.{moving_rate_ptr.*},
                    );
                }
            },
            .producer => {
                zgui.pushIntId(@as(i32, @intCast(popup.hs.id.gui_id)) + 100);
                zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
                zgui.text("Production Rate", .{});
                const production_rate_ptr = &popup.parameters.producer.production_rate;
                if (zgui.sliderScalar("##pr", u32, .{ .v = production_rate_ptr, .min = 1, .max = 1000 })) {
                    const offset = popup.hs.id.gui_id * @sizeOf(Producer) +
                        @offsetOf(Producer, "production_rate");
                    gctx.queue.writeBuffer(
                        gctx.lookupResource(args.producers.buf).?,
                        offset,
                        u32,
                        &.{production_rate_ptr.*},
                    );
                }
                zgui.text("Max Inventory", .{});
                const max_inventory_ptr = &popup.parameters.producer.max_inventory;
                if (zgui.sliderScalar("##mi", u32, .{ .v = max_inventory_ptr, .min = 10, .max = 10000 })) {
                    const offset = popup.hs.id.gui_id * @sizeOf(Producer) +
                        @offsetOf(Producer, "max_inventory");
                    gctx.queue.writeBuffer(
                        gctx.lookupResource(args.producers.buf).?,
                        offset,
                        u32,
                        &.{max_inventory_ptr.*},
                    );
                }
            },
        }
        zgui.popId();
    }
    zgui.end();
}
