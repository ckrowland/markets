const std = @import("std");
const array = std.ArrayList;
const random = std.crypto.random;

const Self = @This();

params: struct {
    num_producers: i32,
    production_rate: i32,
    giving_rate: i32,
    num_consumers: i32,
    consumption_rate: i32,
    moving_rate: f32,
    producer_width: f32,
    consumer_radius: f32,
    num_consumer_sides: u32,
},
coordinate_size: struct {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
},
producers: array(Producer),
consumers: array(Consumer),
stats: struct {
    transactions_array: array(f32),
    num_transactions_last_second: f32,
    previous_second: f32,
    max_transactions_recorded: f32,
},

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .params = .{
            .num_producers = 10,
            .production_rate = 10,
            .giving_rate = 10,
            .num_consumers = 10000,
            .consumption_rate = 10,
            .moving_rate = 5.0,
            .producer_width = 40.0,
            .consumer_radius = 10.0,
            .num_consumer_sides = 20,
        },
        .coordinate_size = .{
            .min_x = -1000,
            .min_y = -1000,
            .max_x = 1000,
            .max_y = 1000,
        },
        .consumers = array(Consumer).init(allocator),
        .producers = array(Producer).init(allocator),
        .stats = .{
            .transactions_array = array(f32).init(allocator),
            .num_transactions_last_second = 0,
            .previous_second = 0,
            .max_transactions_recorded = 0,
        },
    };
}

pub fn deinit(self: *Self) void {
    self.producers.deinit();
    self.consumers.deinit();
    self.stats.transactions_array.deinit();
}

pub fn createAgents(self: *Self) void {
    self.producers.clearAndFree();
    self.consumers.clearAndFree();
    createProducers(self);
    createConsumers(self);
}

pub fn createConsumers(self: *Self) void {
    var i: usize = 0;
    while (i < self.params.num_consumers) {
        const x = @intToFloat(f32, random.intRangeAtMost(i32, self.coordinate_size.min_x, self.coordinate_size.max_x));
        const y = @intToFloat(f32, random.intRangeAtMost(i32, self.coordinate_size.min_y, self.coordinate_size.max_y));
        const pos = @Vector(4, f32){ x, y, 0.0, 0.0 };
        const step_size = @Vector(4, f32){ 0.0, 0.0, 0.0, 0.0 };
        const c = Consumer{
            .position = pos,
            .home = pos,
            .destination = pos,
            .step_size = step_size,
            .consumption_rate = self.params.consumption_rate,
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
        const p = Producer{
            .position = pos,
            .production_rate = self.params.production_rate,
            .giving_rate = self.params.giving_rate,
            .inventory = 1000,
            .width = self.params.producer_width,
        };
        self.producers.append(p) catch unreachable;
        i += 1;
    }
}

pub const Producer = struct {
    position: @Vector(4, f32),
    production_rate: i32,
    giving_rate: i32,
    inventory: i32,
    width: f32,
};

pub const Consumer = struct {
    position: @Vector(4, f32),
    home: @Vector(4, f32),
    destination: @Vector(4, f32),
    step_size: @Vector(4, f32),
    consumption_rate: i32,
    moving_rate: f32,
    inventory: i32,
    radius: f32,
    producer_id: i32,
};
