const std = @import("std");
const Consumers = @import("consumers.zig");
const Consumer = Consumers.Consumer;
const Splines = @import("splines.zig");
const SplinePoint = Splines.SplinePoint;
const array = std.ArrayList;
const random = std.crypto.random;

const Self = @This();

params: struct {
    num_producers: i32,
    production_rate: i32,
    giving_rate: i32,
    max_inventory: i32,
    num_consumers: i32,
    consumption_rate: i32,
    velocity: f32,
    producer_width: f32,
    consumer_radius: f32,
    num_consumer_sides: u32,
},
coordinate_size: CoordinateSize,
consumers: array(Consumer),
stats: Statistics,
asplines: array(SplinePoint),
allocator: std.mem.Allocator,

pub const Statistics = struct {
    num_transactions: array(i32),
    second: i32,
    max_stat_recorded: i32,
    num_empty_consumers: array(i32),
    num_total_producer_inventory: array(i32),
};

pub const CoordinateSize = struct {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    center_x: f32,
    center_y: f32,
};

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .params = .{
            .num_producers = 10,
            .production_rate = 100,
            .giving_rate = 10,
            .max_inventory = 10000,
            .num_consumers = 1000,
            .consumption_rate = 10,
            .velocity = 50.0,
            .producer_width = 40.0,
            .consumer_radius = 10.0,
            .num_consumer_sides = 20,
        },
        .coordinate_size = .{
            .min_x = -1000,
            .min_y = -500,
            .max_x = 1800,
            .max_y = 1200,
            .center_x = 400,
            .center_y = 350,
        },
        .consumers = array(Consumer).init(allocator),
        .stats = .{
            .num_transactions = array(i32).init(allocator),
            .second = 0,
            .max_stat_recorded = 0,
            .num_empty_consumers = array(i32).init(allocator),
            .num_total_producer_inventory = array(i32).init(allocator), 
        },
        .asplines = array(SplinePoint).init(allocator),
        .allocator = allocator,
    };
}

pub fn deinit(self: *Self) void {
    self.consumers.deinit();
    self.asplines.deinit();
    self.stats.num_transactions.deinit();
    self.stats.num_empty_consumers.deinit();
    self.stats.num_total_producer_inventory.deinit();
}

pub fn createAgents(self: *Self) void {
    self.consumers.clearAndFree();
    self.stats.num_transactions.clearAndFree();
    self.stats.num_empty_consumers.clearAndFree();
    self.stats.num_total_producer_inventory.clearAndFree();
    self.stats.num_transactions.append(0) catch unreachable;
    self.stats.num_empty_consumers.append(0) catch unreachable;
    self.stats.num_total_producer_inventory.append(0) catch unreachable;
    self.stats.second = 0;
    self.stats.max_stat_recorded = 0;
    Consumers.createConsumers(self);
    Splines.createSplines(self);
}

pub fn supplyShock(self: *Self) void {
    for (self.consumers.items) |_, i| {
        self.consumers.items[i].inventory = 0;
    }
}
