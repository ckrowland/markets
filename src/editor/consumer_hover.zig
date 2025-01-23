const std = @import("std");
const Consumer = @import("consumer");
const Main = @import("main.zig");

const Self = @This();

absolute_home: [4]i32,
home: [4]f32,
grouping_id: u32 = 0,

pub const z_pos = 0;
pub fn createAndAppend(demo: *Main.DemoState, c: Consumer) void {
    var hovers: [1]Self = .{
        Self{
            .absolute_home = .{ c.absolute_home[0], c.absolute_home[1], z_pos, 1 },
            .home = .{ c.home[0], c.home[1], z_pos, 1 },
            .grouping_id = c.grouping_id,
        },
    };
    const buf = &demo.buffers.data.consumer_hovers;
    demo.gctx.queue.writeBuffer(
        demo.gctx.lookupResource(buf.buf).?,
        buf.mapping.num_structs * @sizeOf(Self),
        Self,
        hovers[0..],
    );
    buf.mapping.num_structs += 1;
}
