const std = @import("std");
const zgpu = @import("zgpu");
const zm = @import("zmath");
const Gctx = zgpu.GraphicsContext;
const wgpu = zgpu.wgpu;
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");

// A mishmash of Wgpu initialization functions and buffer helpers

pub const ObjectBuffer = struct {
    data: zgpu.BufferHandle,
    mapped: zgpu.BufferHandle,
};

// See if you can create generic function
fn StagingBuffer(comptime T: type) type {
    return struct {
        slice: ?[]const T = null,
        buffer: wgpu.Buffer,
        num_structs: u32,
    };
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
};

pub const ComputePipelineInfo = struct {
    cs: [:0]const u8,
    entry_point: [:0]const u8,
};

// Helpers
pub fn getNumStructs(gctx: *zgpu.GraphicsContext, comptime T: type, stat_bufs: ObjectBuffer) u32 {
    const stats = getAll(gctx, u32, .{
        .structs = stat_bufs,
        .num_structs = 8,
    }) catch unreachable;
    switch (T) {
        Consumer => return stats[1],
        Producer => return stats[2],
        else => unreachable,
    }
}

fn GenCallback(comptime T: type) wgpu.BufferMapCallback {
    return struct {
        fn callback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
            const usb = @ptrCast(*StagingBuffer(T), @alignCast(@sizeOf(usize), userdata));
            std.debug.assert(usb.slice == null);
            if (status == .success) {
                usb.slice = usb.buffer.getConstMappedRange(T, 0, usb.num_structs).?;
            } else {
                std.debug.print("[zgpu] Failed to map buffer (code: {any})\n", .{status});
            }
        }
    }.callback;
}

pub const getArgs = struct {
    structs: ObjectBuffer,
    num_structs: u32,
};
pub fn getAll(gctx: *zgpu.GraphicsContext, comptime T: type, args: getArgs) ![]T {
    var buf = StagingBuffer(T){
        .buffer = gctx.lookupResource(args.structs.mapped).?,
        .num_structs = args.num_structs,
    };
    if (buf.num_structs == 0) {
        return error.EmptyBuffer;
    }
    buf.buffer.mapAsync(
        .{ .read = true },
        0,
        @sizeOf(T) * buf.num_structs,
        GenCallback(T),
        @ptrCast(*anyopaque, &buf),
    );
    wait_loop: while (true) {
        gctx.device.tick();
        if (buf.slice == null) {
            continue :wait_loop;
        }
        break;
    }
    buf.buffer.unmap();

    return @constCast(buf.slice.?[0..buf.num_structs]);
}

pub fn getParameters(comptime T: type) type {
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
        else => unreachable,
    }
}

pub fn setArgs(comptime T: type) type {
    switch (T) {
        Consumer => return struct {
            agents: ObjectBuffer,
            stats: ObjectBuffer,
            num_agents: u32,
            parameter: union(enum) {
                moving_rate: f32,
                demand_rate: u32,
            },
        },
        Producer => return struct {
            agents: ObjectBuffer,
            stats: ObjectBuffer,
            num_agents: u32,
            parameter: union(enum) {
                production_rate: u32,
                inventory: i32,
                max_inventory: u32,
            },
        },
        else => unreachable,
    }
}

pub fn setAll(gctx: *zgpu.GraphicsContext, comptime agent: type, args: setArgs(agent)) void {
    var agents = getAll(gctx, agent, .{
        .structs = args.agents,
        .num_structs = getNumStructs(gctx, agent, args.stats),
    }) catch return;
    for (agents, 0..) |_, i| {
        switch (agent) {
            Consumer => {
                switch (args.parameter) {
                    .moving_rate => |v| agents[i].moving_rate = v,
                    .demand_rate => |v| agents[i].demand_rate = v,
                }
            },
            Producer => {
                switch (args.parameter) {
                    .production_rate => |v| agents[i].production_rate = v,
                    .inventory => |v| agents[i].inventory = v,
                    .max_inventory => |v| agents[i].max_inventory = v,
                }
            },
            else => unreachable,
        }
    }
    gctx.queue.writeBuffer(
        gctx.lookupResource(args.agents.data).?,
        0,
        agent,
        agents,
    );
}

pub const shrinkArgs = struct {
    new_size: u32,
    buf: zgpu.BufferHandle,
};
pub fn shrinkBuffer(gctx: *Gctx, comptime T: type, args: shrinkArgs) void {
    const all_zero = [_]u8{0} ** 10000000;
    const buf = gctx.lookupResource(args.buf).?;
    const buf_info = gctx.lookupResourceInfo(args.buf).?;
    const size_to_keep = @sizeOf(T) * args.new_size;
    const size_to_clear = buf_info.size - size_to_keep;
    gctx.queue.writeBuffer(
        buf,
        size_to_keep,
        u8,
        all_zero[0..size_to_clear],
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
    gctx.queue.writeBuffer(
        gctx.lookupResource(buf).?,
        0,
        u8,
        all_zero[0..buf_info.size],
    );
}

// Blank Buffers
pub fn createBuffer(gctx: *Gctx, comptime T: type, num: u32) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .copy_src = true, .vertex = true, .storage = true },
        .size = num * @sizeOf(T),
    });
}

pub fn createMappedBuffer(gctx: *Gctx, comptime T: type, num: u32) zgpu.BufferHandle {
    return gctx.createBuffer(.{
        .usage = .{ .copy_dst = true, .map_read = true },
        .size = num * @sizeOf(T),
    });
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
    producer: zgpu.BufferHandle,
    stats: zgpu.BufferHandle,
};

pub fn createComputeBindGroup(gctx: *Gctx, args: computeBindGroup) zgpu.BindGroupHandle {
    const compute_bgl = createComputeBindGroupLayout(gctx);
    defer gctx.releaseResource(compute_bgl);

    const c_info = gctx.lookupResourceInfo(args.consumer) orelse unreachable;
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
            .buffer_handle = args.producer,
            .offset = 0,
            .size = p_info.size,
        },
        .{
            .binding = 2,
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
            .topology = .triangle_list,
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
