const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");

const Self = @This();

params: struct {
    num_producers: i32 = 10,
    production_rate: i32 = 100,
    giving_rate: i32 = 10,
    max_inventory: i32 = 10000,
    num_consumers: i32 = 10000,
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
producers: array(Producer),
consumers: array(Consumer),
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
        .consumers = array(Consumer).init(allocator),
        .producers = array(Producer).init(allocator),
        .stats = .{
            .num_transactions = array(u32).init(allocator),
            .num_empty_consumers = array(u32).init(allocator),
            .num_total_producer_inventory = array(u32).init(allocator), 
        },
    };
}

pub fn deinit(self: *Self) void {
    self.producers.deinit();
    self.consumers.deinit();
    self.stats.num_transactions.deinit();
    self.stats.num_empty_consumers.deinit();
    self.stats.num_total_producer_inventory.deinit();
}

fn clearSimulation(self: *Self) void {
    self.producers.clearAndFree();
    self.consumers.clearAndFree();
    self.stats.num_transactions.clearAndFree();
    self.stats.num_empty_consumers.clearAndFree();
    self.stats.num_total_producer_inventory.clearAndFree();
    self.stats.num_transactions.append(0) catch unreachable;
    self.stats.num_empty_consumers.append(0) catch unreachable;
    self.stats.num_total_producer_inventory.append(0) catch unreachable;
    self.stats.second = 0;
    self.stats.max_stat_recorded = 0;
}

pub fn createAgents(self: *Self) void {
    clearSimulation(self);
    createProducers(self);
    createConsumers(self);
}

pub fn supplyShock(self: *Self) void {
    for (self.producers.items) |_, i| {
        self.producers.items[i].inventory = 0;
    }
}

pub fn createConsumers(self: *Self) void {
    var i: usize = 0;
    while (i < self.params.num_consumers) {
        const x = @intToFloat(f32, random.intRangeAtMost(i32, self.coordinate_size.min_x, self.coordinate_size.max_x));
        const y = @intToFloat(f32, random.intRangeAtMost(i32, self.coordinate_size.min_y, self.coordinate_size.max_y));
        const pos = @Vector(4, f32){ x, y, 0.0, 0.0 };
        const step_size = @Vector(4, f32){ 0.0, 0.0, 0.0, 0.0 };
        const init_color = @Vector(4, f32){ 0.0, 1.0, 0.0, 0.0 };
        const c = Consumer{
            .position = pos,
            .home = pos,
            .destination = pos,
            .step_size = step_size,
            .color = init_color,
            .moving_rate = self.params.moving_rate,
            .inventory = 0,
            .radius = self.params.consumer_radius,
            .producer_id = 1000,
        };
        self.consumers.append(c) catch unreachable;
        i += 1;
    }
}

pub fn createProducers(self: *Self) void {
    var i: usize = 0;
    while (i < self.params.num_producers) {
        const x = @intToFloat(f32, random.intRangeAtMost(i32, self.coordinate_size.min_x, self.coordinate_size.max_x));
        const y = @intToFloat(f32, random.intRangeAtMost(i32, self.coordinate_size.min_y, self.coordinate_size.max_y));
        const pos = @Vector(4, f32){ x, y, 0.0, 0.0 };
        const init_color = @Vector(4, f32){ 1.0, 1.0, 1.0, 0.0 };
        const q = [_]u32{0} ** 450;
        const p = Producer{
            .position = pos,
            .color = init_color,
            .production_rate = @intCast(u32, self.params.production_rate),
            .giving_rate = @intCast(u32, self.params.giving_rate),
            .inventory = @intCast(u32, self.params.max_inventory),
            .max_inventory = @intCast(u32, self.params.max_inventory),
            .len = 0,
            .queue = q,
        };
        self.producers.append(p) catch unreachable;
        i += 1;
    }
}
