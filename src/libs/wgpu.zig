const std = @import("std");
const zgpu = @import("zgpu");
const zm = @import("zmath");
const Gctx = zgpu.GraphicsContext;
const wgpu = zgpu.wgpu;

pub const MAX_NUM_STRUCTS = 10000;

pub fn CallbackArgs(comptime T: type) type {
    return struct {
        gctx: *zgpu.GraphicsContext,
        obj_buf: *ObjectBuffer(T),
        stat_array: *std.ArrayList(u32) = undefined,
        popup_idx: usize = 0,
    };
}

//pub const GraphicsObject = struct {
//    render_pipeline: zgpu.RenderPipelineHandle,
//    attribute_buffer: zgpu.BufferHandle,
//    vertex_buffer: zgpu.BufferHandle,
//    index_buffer: zgpu.BufferHandle,
//    size_of_struct: u32,
//};

pub fn ObjectBuffer(comptime T: type) type {
    return struct {
        buf: zgpu.BufferHandle,
        mapping: MappingBuffer(T),

        pub fn updateI32Field(
            self: ObjectBuffer(T),
            gctx: *zgpu.GraphicsContext,
            val: i32,
            comptime name: [:0]const u8,
        ) void {
            for (0..self.mapping.num_structs) |i| {
                gctx.queue.writeBuffer(
                    gctx.lookupResource(self.buf).?,
                    i * @sizeOf(T) + @offsetOf(T, name),
                    i32,
                    &.{val},
                );
            }
        }
    };
}

const callback_queue_len: usize = 10;
fn MappingBuffer(comptime T: type) type {
    return struct {
        buf: zgpu.BufferHandle,
        insert_idx: usize = 0,
        remove_idx: usize = 0,
        requests: [callback_queue_len]struct {
            func: Callback(T),
            args: CallbackArgs(T),
        } = undefined,
        staging: StagingBuffer(T),
        state: enum {
            rest,
            copy_to_mapped_buffer,
            call_map_async,
            waiting_for_map_async,
        } = .rest,
        num_structs: u32,
    };
}

fn StagingBuffer(comptime T: type) type {
    return struct {
        slice: ?[]const T = null,
        buffer: wgpu.Buffer = undefined,
        num_structs: u32,
    };
}

fn Callback(comptime T: type) type {
    return ?*const fn (args: CallbackArgs(T)) void;
}

pub const ComputePipelineInfo = struct {
    cs: [:0]const u8,
    entry_point: [:0]const u8,
};

pub fn GenCallback(comptime T: type) wgpu.BufferMapCallback {
    return struct {
        fn callback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
            const usb = @as(*StagingBuffer(T), @ptrCast(@alignCast(userdata)));
            std.debug.assert(usb.slice == null);
            if (status == .success) {
                usb.slice = usb.buffer.getConstMappedRange(T, 0, usb.num_structs).?;
            } else {
                std.log.err("[zgpu] Failed to map buffer (code: {any})\n", .{status});
            }
        }
    }.callback;
}

pub fn getAllAsync(
    comptime T: type,
    callback: Callback(T),
    args: CallbackArgs(T),
) void {
    const map_ptr = &args.obj_buf.mapping;

    map_ptr.staging.num_structs = map_ptr.num_structs;
    if (map_ptr.staging.num_structs <= 0) return;

    map_ptr.requests[map_ptr.insert_idx].func = callback;
    map_ptr.requests[map_ptr.insert_idx].args = args;
    map_ptr.insert_idx = (map_ptr.insert_idx + 1) % callback_queue_len;
    map_ptr.state = .copy_to_mapped_buffer;
}

pub fn checkObjBufState(comptime T: type, buf: *MappingBuffer(T)) void {
    if (buf.state == .call_map_async and buf.staging.slice == null and buf.insert_idx != buf.remove_idx) {
        const gctx = buf.requests[buf.remove_idx].args.gctx;
        buf.staging.buffer = gctx.lookupResource(buf.buf).?;
        buf.staging.buffer.mapAsync(
            .{ .read = true },
            0,
            @sizeOf(T) * buf.staging.num_structs,
            GenCallback(T),
            @as(*anyopaque, @ptrCast(&buf.staging)),
        );
        buf.state = .waiting_for_map_async;
        return;
    }

    if (buf.state == .waiting_for_map_async and buf.staging.slice != null) {
        const request = buf.requests[buf.remove_idx];
        buf.remove_idx = (buf.remove_idx + 1) % callback_queue_len;
        request.func.?(request.args);
        buf.staging.buffer.unmap();
        buf.staging.slice = null;
        buf.state = .rest;
    }
}

pub fn getMappedData(comptime T: type, buf: *MappingBuffer(T)) []T {
    return @constCast(buf.staging.slice.?[0..buf.staging.num_structs]);
}

pub fn writeBuffer(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    comptime T: type,
    structs: []T,
) void {
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, T, structs);
}

pub fn bufArgs(comptime T: type, comptime V: type) type {
    return struct {
        obj_buf: ObjectBuffer(T),
        index: usize,
        value: V,
    };
}
pub fn writeToObjectBuffer(
    gctx: *Gctx,
    comptime T: type,
    comptime V: type,
    comptime field: []const u8,
    args: bufArgs(T, V),
) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.obj_buf.buf).?,
        args.index * @sizeOf(T) + @offsetOf(T, field),
        V,
        &.{args.value},
    );
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.obj_buf.mapping.buf).?,
        args.index * @sizeOf(T) + @offsetOf(T, field),
        V,
        &.{args.value},
    );
}

pub fn writeToMappedBuffer(gctx: *Gctx, buf: zgpu.BufferHandle, mapped: zgpu.BufferHandle) void {
    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        pass: {
            const p = gctx.lookupResource(buf) orelse break :pass;
            const p_info = gctx.lookupResourceInfo(buf) orelse break :pass;
            const pm = gctx.lookupResource(mapped) orelse break :pass;
            const p_size = @as(usize, @intCast(p_info.size));
            encoder.copyBufferToBuffer(p, 0, pm, 0, p_size);
        }
        break :commands encoder.finish(null);
    };
    defer commands.release();
    gctx.submit(&.{commands});
}

pub fn appendArgs(comptime T: type) type {
    return struct {
        num_old_structs: u32,
        buf: zgpu.BufferHandle,
        structs: []T,
    };
}
pub fn appendBuffer(gctx: *Gctx, comptime T: type, args: appendArgs(T)) void {
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.buf).?,
        args.num_old_structs * @sizeOf(T),
        T,
        args.structs,
    );
}

pub fn clearObjBuffer(encoder: wgpu.CommandEncoder, gctx: *Gctx, comptime T: type, obj_buf: *ObjectBuffer(T)) void {
    const buf = gctx.lookupResource(obj_buf.buf).?;
    const map_buf = gctx.lookupResource(obj_buf.mapping.buf).?;
    const buf_info = gctx.lookupResourceInfo(obj_buf.buf).?;
    const map_buf_info = gctx.lookupResourceInfo(obj_buf.mapping.buf).?;

    encoder.clearBuffer(buf, 0, @intCast(buf_info.size));
    encoder.clearBuffer(map_buf, 0, @intCast(map_buf_info.size));

    obj_buf.mapping.insert_idx = 0;
    obj_buf.mapping.remove_idx = 0;
    obj_buf.mapping.state = .rest;
    obj_buf.mapping.staging.slice = null;
    obj_buf.mapping.num_structs = 0;
    obj_buf.mapping.staging.num_structs = 0;
}

// Blank Buffers
pub fn createBuffer(
    gctx: *Gctx,
    comptime T: type,
    num: u32,
) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .vertex = true, .storage = true },
        .size = num * @sizeOf(T),
    });
}

pub fn createMappedBuffer(
    gctx: *Gctx,
    comptime T: type,
    num: u32,
) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = num * @sizeOf(T),
    });
}

pub fn createObjectBuffer(
    gctx: *Gctx,
    comptime T: type,
    len: u32,
    num_structs: u32,
) ObjectBuffer(T) {
    return .{
        .buf = createBuffer(gctx, T, len),
        .mapping = .{
            .buf = createMappedBuffer(gctx, T, len),
            .num_structs = num_structs,
            .staging = .{
                .num_structs = num_structs,
            },
        },
    };
}

// Depth Texture
pub const Depth = struct {
    texture: zgpu.TextureHandle,
    view: zgpu.TextureViewHandle,
};
pub fn createDepthTexture(gctx: *zgpu.GraphicsContext) Depth {
    const texture = gctx.createTexture(.{
        .usage = .{ .render_attachment = true },
        .dimension = .tdim_2d,
        .size = .{
            .width = gctx.swapchain_descriptor.width,
            .height = gctx.swapchain_descriptor.height,
            .depth_or_array_layers = 1,
        },
        .format = .depth24_plus,
        .mip_level_count = 1,
        .sample_count = 1,
    });
    const view = gctx.createTextureView(texture, .{});
    return .{ .texture = texture, .view = view };
}

// Bind Group Layouts
pub fn createComputeBindGroupLayout(gctx: *Gctx) zgpu.BindGroupLayoutHandle {
    return gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(1, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(2, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(3, .{ .compute = true }, .storage, false, 0),
    });
}

// Bind Groups
pub const computeBindGroup = struct {
    consumer: zgpu.BufferHandle,
    consumer_params: zgpu.BufferHandle,
    producer: zgpu.BufferHandle,
    stats: zgpu.BufferHandle,
};

pub fn createComputeBindGroup(gctx: *Gctx, args: computeBindGroup) zgpu.BindGroupHandle {
    const compute_bgl = createComputeBindGroupLayout(gctx);
    defer gctx.releaseResource(compute_bgl);

    const c_info = gctx.lookupResourceInfo(args.consumer) orelse unreachable;
    const cp_info = gctx.lookupResourceInfo(args.consumer_params) orelse unreachable;
    const p_info = gctx.lookupResourceInfo(args.producer) orelse unreachable;
    const s_info = gctx.lookupResourceInfo(args.stats) orelse unreachable;

    return gctx.createBindGroup(compute_bgl, &[_]zgpu.BindGroupEntryInfo{
        .{
            .binding = 0,
            .buffer_handle = args.consumer,
            .offset = 0,
            .size = c_info.size,
        },
        .{
            .binding = 1,
            .buffer_handle = args.consumer_params,
            .offset = 0,
            .size = cp_info.size,
        },
        .{
            .binding = 2,
            .buffer_handle = args.producer,
            .offset = 0,
            .size = p_info.size,
        },
        .{
            .binding = 3,
            .buffer_handle = args.stats,
            .offset = 0,
            .size = s_info.size,
        },
    });
}

fn getWgpuType(comptime T: type) !wgpu.VertexFormat {
    return switch (T) {
        u32 => .uint32,
        f32 => .float32,
        [2]f32 => .float32x2,
        [3]f32 => .float32x3,
        [4]f32 => .float32x4,
        else => error.NoValidWgpuType,
    };
}

pub const RenderPipelineInfo = struct {
    inst_type: type,
    inst_attrs: []const struct {
        name: []const u8,
        type: type,
    },
    primitive_topology: wgpu.PrimitiveTopology = .triangle_list,
    vs: [:0]const u8,
};

pub fn createRenderPipeline(
    gctx: *zgpu.GraphicsContext,
    bind_group_layout: zgpu.BindGroupLayoutHandle,
    comptime args: RenderPipelineInfo,
) zgpu.RenderPipelineHandle {
    const vs_module = zgpu.createWgslShaderModule(gctx.device, args.vs, "vs");
    defer vs_module.release();

    const color_targets = [_]wgpu.ColorTargetState{.{
        .format = zgpu.GraphicsContext.swapchain_format,
        .blend = &.{ .color = .{}, .alpha = .{} },
    }};

    const vertex_attributes = [_]wgpu.VertexAttribute{
        .{ .format = .float32x3, .offset = 0, .shader_location = 0 },
    };

    const instance_attributes = init: {
        var arr: [args.inst_attrs.len]wgpu.VertexAttribute = undefined;
        inline for (args.inst_attrs, 0..) |attr, i| {
            arr[i] = .{
                .format = getWgpuType(attr.type) catch unreachable,
                .offset = @offsetOf(args.inst_type, attr.name),
                .shader_location = i + 1,
            };
        }
        break :init arr;
    };

    const vertex_buffers = [_]wgpu.VertexBufferLayout{
        .{
            .array_stride = @sizeOf(f32) * 3,
            .attribute_count = vertex_attributes.len,
            .attributes = &vertex_attributes,
            .step_mode = .vertex,
        },
        .{
            .array_stride = @sizeOf(args.inst_type),
            .attribute_count = instance_attributes.len,
            .attributes = &instance_attributes,
            .step_mode = .instance,
        },
    };

    const pipeline_descriptor = wgpu.RenderPipelineDescriptor{
        .vertex = wgpu.VertexState{
            .module = vs_module,
            .entry_point = "vs",
            .buffer_count = vertex_buffers.len,
            .buffers = &vertex_buffers,
        },
        .primitive = wgpu.PrimitiveState{
            .front_face = .ccw,
            .cull_mode = .none,
            .topology = args.primitive_topology,
        },
        .depth_stencil = &wgpu.DepthStencilState{
            .format = .depth24_plus,
            .depth_write_enabled = true,
            .depth_compare = .less_equal,
        },
        .fragment = &wgpu.FragmentState{
            .module = vs_module,
            .entry_point = "fs",
            .target_count = color_targets.len,
            .targets = &color_targets,
        },
    };

    const pipeline_layout = gctx.createPipelineLayout(&.{bind_group_layout});
    return gctx.createRenderPipeline(pipeline_layout, pipeline_descriptor);
}

pub fn createComputePipeline(gctx: *zgpu.GraphicsContext, cpi: ComputePipelineInfo) zgpu.ComputePipelineHandle {
    const compute_bgl = createComputeBindGroupLayout(gctx);
    defer gctx.releaseResource(compute_bgl);

    const compute_pl = gctx.createPipelineLayout(&.{compute_bgl});
    defer gctx.releaseResource(compute_pl);

    const cs_module = zgpu.createWgslShaderModule(gctx.device, cpi.cs, "cs");
    defer cs_module.release();

    const pipeline_descriptor = wgpu.ComputePipelineDescriptor{
        .compute = wgpu.ProgrammableStageDescriptor{
            .module = cs_module,
            .entry_point = cpi.entry_point,
        },
    };

    return gctx.createComputePipeline(compute_pl, pipeline_descriptor);
}
