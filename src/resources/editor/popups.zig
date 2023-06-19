const std = @import("std");
const zgui = @import("zgui");

const Window = struct {
    x: f32,
    y: f32,
    width_percent: f32 = 0.3,
    height_percent: f32 = 0.3,
    flags: zgui.WindowFlags = .{},
};

pub const window_flags = .{
    .popen = null,
    .flags = zgui.WindowFlags.no_decoration,
};

const Self = @This();
positions: std.ArrayList(Window),

pub fn init(allocator: std.mem.Allocator) Self {
    return Self{
        .positions = std.ArrayList(Window).init(allocator),
    };
}

pub fn deinit(self: Self) void {
    self.positions.deinit();
}

pub fn addWindow(self: *Self, window: Window) void {
    self.positions.append(window) catch unreachable;
}

pub fn display(self: Self) void {
    _ = self;
}    
// pub fn producerParameters() void {
//     if (zgui.begin("Test", Window.window_flags)) {
//         zgui.pushIntId(3);

//         zgui.text("Production Rate", .{});
//         if (zgui.sliderScalar(
//             "##pr",
//             u32,
//             .{ .v = &demo.params.production_rate, .min = 1, .max = 1000 },
//         )) {
//             Wgpu.setAll(gctx, Producer, .{
//                 .get_buffer = demo.buffers.data.producer,
//                 .stats = demo.buffers.data.stats,
//                 .num_agents = demo.params.num_producers.new,
//                 .parameter = .{
//                     .production_rate = demo.params.production_rate,
//                 },
//             });
//         }
        
//         zgui.popId();
//     }
//     zgui.end();
// }    
