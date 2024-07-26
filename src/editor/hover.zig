const zgpu = @import("zgpu");
const Self = @This();

position: [4]f32,
color: [4]f32,
radius: f32,
pub const z_pos = 0;
pub fn initBuffer(gctx: *zgpu.GraphicsContext) zgpu.BufferHandle {
    const buf = gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .vertex = true, .storage = true },
        .size = @sizeOf(Self),
    });
    const hoverCircle = Self{
        .position = .{ 0, 0, z_pos, 0 },
        .color = .{ 0, 0, 1, 0 },
        .radius = 300,
    };
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        Self,
        &.{hoverCircle},
    );
    return buf;
}
