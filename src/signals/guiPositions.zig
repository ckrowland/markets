const Window = @import("../windows.zig");

pub const plot = Window.Args{
    .x = 0.25,
    .y = 0.0,
    .w = 0.75,
    .h = 1.0,
    .margin = 0.02,
    .no_margin = .{
        .left = true,
    },
};

pub const parameter = Window.Args{
    .x = 0.0,
    .y = 0.13,
    .w = 0.25,
    .h = 0.62,
    .margin = 0.02,
};
