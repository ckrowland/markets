const zgpu = @import("zgpu");
const zstbi = @import("zstbi");

pub fn createTextureView(gctx: *zgpu.GraphicsContext, path: [:0]const u8) !zgpu.TextureViewHandle {
    var image = try zstbi.Image.loadFromFile(path, 4);
    defer image.deinit();

    const texture = gctx.createTexture(.{
        .usage = .{ .texture_binding = true, .copy_dst = true },
        .size = .{
            .width = image.width,
            .height = image.height,
            .depth_or_array_layers = 1,
        },
        .format = zgpu.imageInfoToTextureFormat(
            image.num_components,
            image.bytes_per_component,
            image.is_hdr,
        ),
        .mip_level_count = 1,
    });
    const textureView = gctx.createTextureView(texture, .{});

    gctx.queue.writeTexture(
        .{ .texture = gctx.lookupResource(texture).? },
        .{
            .bytes_per_row = image.bytes_per_row,
            .rows_per_image = image.height,
        },
        .{ .width = image.width, .height = image.height },
        u8,
        image.data,
    );

    return textureView;
}
