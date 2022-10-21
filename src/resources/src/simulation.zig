const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");

const Self = @This();

params: struct {
    num_producers: u32 = 10,
    production_rate: u32 = 100,
    giving_rate: u32 = 10,
    max_inventory: u32 = 10000,
    num_consumers: u32 = 10000,
    moving_rate: f32 = 5.0,
    producer_width: f32 = 40.0,
    consumer_radius: f32 = 10.0,
    num_consumer_sides: u32 = 20,
},
coordinate_size: struct {
    min_x: i32 = -1000,
    min_y: i32 = -500,
    max_x: i32 = 1800,
    max_y: i32 = 1200,
},
stats: struct {
    num_transactions: array(u32),
    second: f32 = 0,
    max_stat_recorded: u32 = 0,
    num_empty_consumers: array(u32),
    num_total_producer_inventory: array(u32),
},

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .params = .{},
        .coordinate_size = .{},
        .stats = .{
            .num_transactions = array(u32).init(allocator),
            .num_empty_consumers = array(u32).init(allocator),
            .num_total_producer_inventory = array(u32).init(allocator), 
        },
    };
}

pub fn deinit(self: *Self) void {
    self.stats.num_transactions.deinit();
    self.stats.num_empty_consumers.deinit();
    self.stats.num_total_producer_inventory.deinit();
}

fn clearSimulation(self: *Self) void {
    self.stats.num_transactions.clearAndFree();
    self.stats.num_empty_consumers.clearAndFree();
    self.stats.num_total_producer_inventory.clearAndFree();
    self.stats.num_transactions.append(0) catch unreachable;
    self.stats.num_empty_consumers.append(0) catch unreachable;
    self.stats.num_total_producer_inventory.append(0) catch unreachable;
    self.stats.second = 0;
    self.stats.max_stat_recorded = 0;
}
