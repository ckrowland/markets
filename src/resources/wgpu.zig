const std = @import("std");
const zgpu = @import("zgpu");
const zm = @import("zmath");
const Gctx = zgpu.GraphicsContext;
const wgpu = zgpu.wgpu;
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const ConsumerHover = @import("editor/consumer_hover.zig");
const Camera = @import("../camera.zig");

pub const MAX_NUM_STRUCTS = 10000;

// A mishmash of Wgpu initialization functions and buffer helpers for an array of generic structs

// Data Types
pub const GraphicsObject = struct {
    render_pipeline: zgpu.RenderPipelineHandle,
    attribute_buffer: zgpu.BufferHandle,
    vertex_buffer: zgpu.BufferHandle,
    index_buffer: zgpu.BufferHandle,
    size_of_struct: u32,
};
//    pub const Resources = struct {
//        render_pipeline: wgpu.RenderPipeline,
//        attribute_buffer: zgpu.BufferInfo,
//        vertex_buffer: zgpu.BufferInfo,
//        index_buffer: zgpu.BufferInfo,
//    };
//    pub fn getResources(
//        self: GraphicsObject,
//        gctx: *zgpu.GraphicsContext,
//    ) !Resources {
//        return .{
//            .render_pipeline = try gctx.lookupResource(self.render_pipeline),
//            .attribute_buffer = try gctx.lookupResourceInfo(self.attribute_buffer),
//            .vertex_buffer = try gctx.lookupResourceInfo(self.vertex_buffer),
//            .index_buffer = try gctx.lookupResourceInfo(self.index_buffer),
//        };
//    }
pub fn draw(
    obj: GraphicsObject,
    gctx: *zgpu.GraphicsContext,
    pass: wgpu.RenderPassEncoder,
) void {
    const render_pipeline = gctx.lookupResource(obj.render_pipeline) orelse return;
    const vertex_buffer = gctx.lookupResourceInfo(obj.vertex_buffer) orelse return;
    const attribute_buffer = gctx.lookupResourceInfo(obj.attribute_buffer) orelse return;
    const index_buffer = gctx.lookupResourceInfo(obj.index_buffer) orelse return;

    pass.setPipeline(render_pipeline);
    pass.setVertexBuffer(
        0,
        vertex_buffer.gpuobj.?,
        0,
        vertex_buffer.size,
    );
    pass.setVertexBuffer(
        1,
        attribute_buffer.gpuobj.?,
        0,
        attribute_buffer.size,
    );
    pass.setIndexBuffer(
        index_buffer.gpuobj.?,
        .uint32,
        0,
        index_buffer.size,
    );

    const num_indices: u32 = @intCast(index_buffer.size / @sizeOf(u32));
    //const num_structs: u32 = @intCast(attribute_buffer.size / obj.size_of_struct);
    pass.drawIndexed(num_indices, 1, 0, 0, 0);
}

pub const ObjectBuffer = struct {
    data: zgpu.BufferHandle,
    mapped: zgpu.BufferHandle,
};

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
    primitive_topology: wgpu.PrimitiveTopology = .triangle_list,
};

pub const ComputePipelineInfo = struct {
    cs: [:0]const u8,
    entry_point: [:0]const u8,
};

// Functions
pub fn getNumStructs(gctx: *zgpu.GraphicsContext, comptime T: type, stat_bufs: ObjectBuffer) u32 {
    const stats = getAll(gctx, u32, .{
        .structs = stat_bufs,
        .num_structs = 8,
    }) catch unreachable;
    switch (T) {
        Consumer => return stats[1],
        Producer => return stats[2],
        ConsumerHover => return stats[3],
        else => unreachable,
    }
}

fn GenCallback(comptime T: type) wgpu.BufferMapCallback {
    return struct {
        fn callback(status: wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.C) void {
            const usb = @as(*StagingBuffer(T), @ptrCast(@alignCast(userdata)));
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
        @as(*anyopaque, @ptrCast(&buf)),
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

pub fn getLast(gctx: *zgpu.GraphicsContext, comptime T: type, args: getArgs) !T {
    const structs = try getAll(gctx, T, args);
    return structs[args.num_structs - 1];
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
        ConsumerHover => return union(enum) {
            color: [4]f32,
        },
        else => unreachable,
    }
}
pub fn setArgs(comptime T: type) type {
    return struct {
        agents: ObjectBuffer,
        num_structs: u32,
        parameter: agentParameters(T),
    };
}
pub fn setAll(gctx: *zgpu.GraphicsContext, comptime T: type, args: setArgs(T)) void {
    var agents = getAll(gctx, T, .{
        .structs = args.agents,
        .num_structs = args.num_structs,
    }) catch return;
    for (agents, 0..) |_, i| {
        setAgentParameter(T, &agents[i], args.parameter);
    }
    writeBuffer(gctx, args.agents.data, T, agents);
}

pub fn writeBuffer(
    gctx: *zgpu.GraphicsContext,
    buf: zgpu.BufferHandle,
    comptime T: type,
    structs: []T,
) void {
    gctx.queue.writeBuffer(gctx.lookupResource(buf).?, 0, T, structs);
}
pub fn setAgentArgs(comptime T: type) type {
    return struct {
        setArgs: setArgs(T),
        grid_pos: [2]i32,
    };
}
pub fn setAgent(gctx: *zgpu.GraphicsContext, comptime T: type, args: setAgentArgs(T)) void {
    var agents = getAll(gctx, T, .{
        .structs = args.setArgs.agents,
        .num_structs = args.setArgs.num_structs,
    }) catch return;
    for (agents, 0..) |agent, i| {
        const grid_pos = agent.absolute_home;
        if (args.grid_pos[0] == grid_pos[0] and args.grid_pos[1] == grid_pos[1]) {
            setAgentParameter(T, &agents[i], args.setArgs.parameter);
        }
    }
    writeBuffer(gctx, args.setArgs.agents.data, T, agents);
}

fn setAgentParameter(comptime T: type, agent: *T, parameter: agentParameters(T)) void {
    switch (T) {
        Consumer => {
            switch (parameter) {
                .moving_rate => |v| agent.moving_rate = v,
                .demand_rate => |v| agent.demand_rate = v,
            }
        },
        Producer => {
            switch (parameter) {
                .production_rate => |v| agent.production_rate = v,
                .inventory => |v| agent.inventory = v,
                .max_inventory => |v| agent.max_inventory = v,
            }
        },
        ConsumerHover => {
            switch (parameter) {
                .color => |v| agent.color = v,
            }
        },
        else => unreachable,
    }
}

pub fn setGroupingArgs(comptime T: type) type {
    return struct {
        setArgs: setArgs(T),
        grouping_id: u32,
    };
}
pub fn setGroup(gctx: *zgpu.GraphicsContext, comptime T: type, args: setGroupingArgs(T)) void {
    var agents = getAll(gctx, T, .{
        .structs = args.setArgs.agents,
        .num_structs = args.setArgs.num_structs,
    }) catch return;
    for (agents, 0..) |agent, i| {
        if (args.grouping_id == agent.grouping_id) {
            setAgentParameter(T, &agents[i], args.setArgs.parameter);
        }
    }
    writeBuffer(gctx, args.setArgs.agents.data, T, agents);
}

pub const updateCoordArgs = struct {
    structs: ObjectBuffer,
    stats: ObjectBuffer,
};
pub fn updateCoords(gctx: *zgpu.GraphicsContext, comptime T: type, args: updateCoordArgs) void {
    const structs = getAll(gctx, T, .{
        .structs = args.structs,
        .num_structs = getNumStructs(gctx, T, args.stats),
    }) catch return;
    var new_structs: [MAX_NUM_STRUCTS]T = undefined;
    for (structs, 0..) |s, i| {
        const world_pos = Camera.getWorldPosition(gctx, s.absolute_home);
        new_structs[i] = s;
        switch (T) {
            Consumer => {
                new_structs[i].position = world_pos;
                new_structs[i].home = world_pos;
                new_structs[i].destination = world_pos;
            },
            Producer => {
                new_structs[i].home = world_pos;
            },
            ConsumerHover => {
                new_structs[i].home = world_pos;
            },
            else => unreachable,
        }
    }

    gctx.queue.writeBuffer(
        gctx.lookupResource(args.structs.data).?,
        0,
        T,
        new_structs[0..structs.len],
    );

    // Since aspect update is done at end of draw loop,
    // updateCoords must write to the mapped buffers before next update
    writeToMappedBuffer(gctx, args.structs);
}

pub fn writeToMappedBuffer(gctx: *Gctx, obj_buf: ObjectBuffer) void {
    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        pass: {
            const p = gctx.lookupResource(obj_buf.data) orelse break :pass;
            const p_info = gctx.lookupResourceInfo(obj_buf.data) orelse break :pass;
            const pm = gctx.lookupResource(obj_buf.mapped) orelse break :pass;
            encoder.copyBufferToBuffer(p, 0, pm, 0, p_info.size);
        }
        break :commands encoder.finish(null);
    };
    defer commands.release();
    gctx.submit(&.{commands});
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

pub fn createObjectBuffer(gctx: *Gctx, comptime T: type, num: u32) ObjectBuffer {
    return .{
        .data = createBuffer(gctx, T, num),
        .mapped = createMappedBuffer(gctx, T, num),
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
