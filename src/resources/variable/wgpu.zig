const std = @import("std");
const zgpu = @import("zgpu");
const zm = @import("zmath");
const zems = @import("zems");
const Gctx = zgpu.GraphicsContext;
const wgpu = zgpu.wgpu;
const DemoState = @import("main.zig").DemoState;
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Camera = @import("camera.zig");
const Statistics = @import("statistics.zig");
const Callbacks = @import("callbacks.zig");

pub const MAX_NUM_STRUCTS = 10000;

// A mishmash of Wgpu initialization functions and buffer helpers for an array of generic structs
// Data Types
pub fn ObjectBuffer(comptime T: type) type {
    return struct {
        buf: zgpu.BufferHandle,
        mapping: MappingBuffer(T),
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
            args: Callbacks.Args(T),
        } = undefined,
        staging: StagingBuffer(T),
        waiting: bool = false,
        num_structs: u32,
    };
}

fn StagingBuffer(comptime T: type) type {
    return struct {
        slice: ?[]const T = null,
        buffer: wgpu.Buffer = undefined,
        num_structs: u32,
        ready: bool = false,
    };
}

fn Callback(comptime T: type) type {
    return ?*const fn (args: Callbacks.Args(T)) void;
}

pub const RenderPipelineInfo = struct {
    pub const Attribute = struct {
        name: []const u8,
        type: type,
    };

    vs: [:0]const u8,
    fs: [:0]const u8,
    inst_type: type,
    inst_attrs: []const Attribute,
    primitive_topology: wgpu.PrimitiveTopology = .triangle_list,
};

pub const ComputePipelineInfo = struct {
    cs: [:0]const u8,
    entry_point: [:0]const u8,
};

pub fn setObjBufField(
    gctx: *zgpu.GraphicsContext,
    comptime Object: type,
    comptime tag: []const u8,
    value: anytype,
    obj_buf: ObjectBuffer(Object),
) void {
    const resource = gctx.lookupResource(obj_buf.buf).?;
    const fieldI = std.meta.fieldIndex(Object, tag);
    const fieldType = @typeInfo(Object).Struct.fields[fieldI.?].type;
    const field_offset = @offsetOf(Object, tag);
    for (0..obj_buf.mapping.num_structs) |i| {
        const offset = i * @sizeOf(Object) + field_offset;
        gctx.queue.writeBuffer(resource, offset, fieldType, &.{value});
    }
}

pub fn GenCallback(comptime T: type) wgpu.BufferMapCallback {
    return struct {
        fn callback(
            status: wgpu.BufferMapAsyncStatus,
            userdata: ?*anyopaque,
        ) callconv(.C) void {
            const usb = @as(*StagingBuffer(T), @ptrCast(@alignCast(userdata)));
            std.debug.assert(usb.slice == null);
            if (status == .success) {
                const data = usb.buffer.getConstMappedRange(
                    T,
                    0,
                    usb.num_structs,
                );
                if (data) |d| {
                    usb.slice = d;
                }
                usb.ready = true;
            } else {
                std.log.err("[zgpu] Failed to map buffer (code: {any})\n", .{status});
            }
        }
    }.callback;
}

pub fn getAllAsync(
    demo: *DemoState,
    comptime T: type,
    callback: Callback(T),
) void {
    const buf = switch (T) {
        u32 => &demo.buffers.data.stats,
        Consumer => &demo.buffers.data.consumers,
        Producer => &demo.buffers.data.producers,
        else => unreachable,
    };

    const map_ptr = &buf.mapping;
    if (map_ptr.num_structs < 0) return;
    map_ptr.staging.num_structs = map_ptr.num_structs;
    map_ptr.requests[map_ptr.insert_idx].func = callback;
    map_ptr.requests[map_ptr.insert_idx].args = .{ .gctx = demo.gctx, .buf = buf, .stats = &demo.stats };
    map_ptr.insert_idx = (map_ptr.insert_idx + 1) % callback_queue_len;

    runMapIfReady(T, map_ptr);
}

pub fn runMapIfReady(comptime T: type, buf: *MappingBuffer(T)) void {
    if (!buf.waiting and buf.staging.slice == null and buf.insert_idx != buf.remove_idx) {
        const gctx = buf.requests[buf.remove_idx].args.gctx;
        buf.staging.buffer = gctx.lookupResource(buf.buf).?;
        buf.staging.buffer.mapAsync(
            .{ .read = true },
            0,
            @sizeOf(T) * buf.staging.num_structs,
            GenCallback(T),
            @as(*anyopaque, @ptrCast(&buf.staging)),
        );
        buf.waiting = true;
    }
}

pub fn runCallbackIfReady(comptime T: type, buf: *MappingBuffer(T)) void {
    const request = buf.requests[buf.remove_idx];
    if (buf.waiting and buf.staging.ready) {
        buf.remove_idx = (buf.remove_idx + 1) % callback_queue_len;
        buf.staging.buffer.unmap();
        request.func.?(request.args);
        buf.staging.slice = null;
        buf.waiting = false;
        buf.staging.ready = false;
    }
}

pub fn waitForCallback(comptime T: type, buf: *MappingBuffer(T)) void {
    while (buf.waiting) {
        runCallbackIfReady(T, buf);
    }
}

pub fn getMappedData(comptime T: type, buf: *MappingBuffer(T)) []T {
    return @constCast(buf.staging.slice.?[0..buf.staging.num_structs]);
}

pub fn agentParameters(comptime T: type) type {
    switch (T) {
        Consumer => return union(enum) {
            moving_rate: f32,
            demand_rate: u32,
        },
        Producer => return union(enum) {
            production_rate: u32,
            inventory: i32,
            max_inventory: u32,
        },
        u32 => return u32,
        else => unreachable,
    }
}
pub fn setArgs(comptime T: type) type {
    return struct {
        agents: ObjectBuffer,
        parameter: agentParameters(T),
    };
}

pub fn writeBuffer(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    comptime T: type,
    structs: []T,
) void {
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, T, structs);
}

pub fn shrinkBuffer(
    gctx: *Gctx,
    buf: zgpu.BufferHandle,
    comptime T: type,
    new_size: u32,
) void {
    const all_zero = [_]u8{0} ** 10000000;
    const buff = gctx.lookupResource(buf).?;
    const buf_info = gctx.lookupResourceInfo(buf).?;
    const size_to_keep = @sizeOf(T) * new_size;
    const size_to_clear = buf_info.size - size_to_keep;
    const usize_to_clear = @as(usize, @intCast(size_to_clear));
    gctx.queue.writeBuffer(
        buff,
        size_to_keep,
        u8,
        all_zero[0..usize_to_clear],
    );
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

pub fn clearBuffer(gctx: *Gctx, buf: zgpu.BufferHandle) void {
    const all_zero = [_]u8{0} ** 10000000;
    const buf_info = gctx.lookupResourceInfo(buf).?;
    const b_size = @as(usize, @intCast(buf_info.size));
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        u8,
        all_zero[0..b_size],
    );
}

pub fn clearObjBuffer(gctx: *Gctx, comptime T: type, obj_buf: *ObjectBuffer(T)) void {
    const all_zero = [_]u8{0} ** 10000000;
    const buf_info = gctx.lookupResourceInfo(obj_buf.buf).?;
    const b_size = @as(usize, @intCast(buf_info.size));
    gctx.queue.writeBuffer(
        gctx.lookupResource(obj_buf.buf).?,
        0,
        u8,
        all_zero[0..b_size],
    );

    const map_buf_info = gctx.lookupResourceInfo(obj_buf.mapping.buf).?;
    const m_size = @as(usize, @intCast(map_buf_info.size));
    gctx.queue.writeBuffer(
        gctx.lookupResource(obj_buf.mapping.buf).?,
        0,
        u8,
        all_zero[0..m_size],
    );

    obj_buf.mapping.insert_idx = 0;
    obj_buf.mapping.remove_idx = 0;
    obj_buf.mapping.waiting = false;
    obj_buf.mapping.staging.ready = false;
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
        .format = .depth32_float,
        .mip_level_count = 1,
        .sample_count = 1,
    });
    const view = gctx.createTextureView(texture, .{});
    return .{ .texture = texture, .view = view };
}

// Bind Group Layouts
pub fn createUniformBindGroupLayout(gctx: *Gctx) zgpu.BindGroupLayoutHandle {
    return gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
    });
}

pub fn createComputeBindGroupLayout(gctx: *Gctx) zgpu.BindGroupLayoutHandle {
    return gctx.createBindGroupLayout(&.{
        zgpu.bufferEntry(0, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(1, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(2, .{ .compute = true }, .storage, false, 0),
        zgpu.bufferEntry(3, .{ .compute = true }, .storage, false, 0),
    });
}

// Bind Groups
pub fn createUniformBindGroup(gctx: *Gctx) zgpu.BindGroupHandle {
    const bind_group_layout = createUniformBindGroupLayout(gctx);
    defer gctx.releaseResource(bind_group_layout);

    return gctx.createBindGroup(bind_group_layout, &.{
        .{ .binding = 0, .buffer_handle = gctx.uniforms.buffer, .offset = 0, .size = @sizeOf(zm.Mat) },
    });
}

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

pub fn createRenderPipeline(
    gctx: *zgpu.GraphicsContext,
    comptime args: RenderPipelineInfo,
) zgpu.RenderPipelineHandle {
    const vs_module = zgpu.createWgslShaderModule(gctx.device, args.vs, "vs");
    defer vs_module.release();

    const fs_module = zgpu.createWgslShaderModule(gctx.device, args.fs, "fs");
    defer fs_module.release();

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
            .entry_point = "main",
            .buffer_count = vertex_buffers.len,
            .buffers = &vertex_buffers,
        },
        .primitive = wgpu.PrimitiveState{
            .front_face = .ccw,
            .cull_mode = .none,
            .topology = args.primitive_topology,
        },
        .depth_stencil = &wgpu.DepthStencilState{
            .format = .depth32_float,
            .depth_write_enabled = true,
            .depth_compare = .less_equal,
        },
        .fragment = &wgpu.FragmentState{
            .module = fs_module,
            .entry_point = "main",
            .target_count = color_targets.len,
            .targets = &color_targets,
        },
    };

    const bind_group_layout = createUniformBindGroupLayout(gctx);
    defer gctx.releaseResource(bind_group_layout);

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
