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

pub fn init() Self {
    return Self{
        .params = .{},
        .coordinate_size = .{},
    };
}
