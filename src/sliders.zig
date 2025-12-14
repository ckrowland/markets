pub fn Slider(comptime T: type) type {
    return struct {
        val: T,
        min: T,
        max: T,
        old: ?T = null,
        restart: ?T = null,
        help: [:0]const u8 = "",
        field_name: [:0]const u8 = "",
        slider_name: [:0]const u8 = "",
    };
}

pub const Parameters = struct {
    num_producers: Slider(u32) = Slider(u32){
        .min = 1,
        .val = 1,
        .old = 1,
        .restart = 1,
        .max = 20,
    },
    production_cost: Slider(u32) = Slider(u32){
        .min = 1,
        .val = 1,
        .max = 5,
        .help = "The amount of money a Producer is required to spend to create a resource.",
    },
    price: Slider(u32) = Slider(u32){
        .min = 1,
        .val = 2,
        .restart = 2,
        .max = 5000,
        .help = "The amount of money a consumer must spend to purchase a resource.",
    },
    max_production_rate: Slider(u32) = Slider(u32){
        .min = 1,
        .val = 100,
        .max = 3000,
        .help = "The maximum amount of resources a Producer can produce at once.",
    },
    max_inventory: Slider(u32) = Slider(u32){
        .min = 5000,
        .val = 5000,
        .max = 20000,
        .help = "The maximum amount of resources a Producer can hold.",
    },
    max_producer_money: Slider(u32) = Slider(u32){
        .min = 5000,
        .val = 20000,
        .max = 40000,
        .help = "The maximum amount of money a Producer can hold.",
    },
    decay_rate: Slider(u32) = Slider(u32){
        .min = 0,
        .val = 0,
        .max = 100,
        .help = "The amount of inventory that spoils in a Producers inventory every frame.",
    },

    num_consumers: Slider(u32) = Slider(u32){
        .min = 1,
        .val = 10,
        .old = 10,
        .restart = 10,
        .max = 5000,
    },
    income: Slider(u32) = Slider(u32){
        .min = 0,
        .val = 20,
        .max = 20,
        .help = "The amount of money a consumer receives every frame.",
    },
    max_consumer_money: Slider(u32) = Slider(u32){
        .min = 1000,
        .val = 1000,
        .max = 20000,
        .help = "The maximum amount of money a Consumer can hold.",
    },
    moving_rate: Slider(f32) = Slider(f32){
        .min = 1,
        .val = 8,
        .max = 50,
    },
    consumer_size: Slider(f32) = Slider(f32){
        .min = 1,
        .val = 10,
        .max = 20,
    },
    producer_size: Slider(f32) = Slider(f32){
        .min = 1,
        .val = 5,
        .max = 20,
    },
    num_consumer_sides: u32 = 20,
    aspect: f32,
};
