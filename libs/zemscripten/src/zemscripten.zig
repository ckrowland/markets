//! Zig bindings and glue for Emscripten

const std = @import("std");
pub const is_emscripten = true;

comptime {
    _ = std.testing.refAllDeclsRecursive(@This());
}

extern fn emscripten_err([*c]const u8) void;
extern fn emscripten_console_error([*c]const u8) void;
extern fn emscripten_console_warn([*c]const u8) void;
extern fn emscripten_console_log([*c]const u8) void;

extern fn emmalloc_memalign(u32, u32) ?*anyopaque;
extern fn emmalloc_realloc_try(?*anyopaque, u32) ?*anyopaque;
extern fn emmalloc_free(?*anyopaque) void;

pub extern fn emscripten_sleep(ms: u32) void;

pub const MainLoopCallback = *const fn () callconv(.C) void;
extern fn emscripten_set_main_loop(MainLoopCallback, c_int, c_int) void;
//void emscripten_set_main_loop(em_callback_func func, int fps, bool simulate_infinite_loop);
pub fn setMainLoop(cb: MainLoopCallback, maybe_fps: ?i16, simulate_infinite_loop: bool) void {
    emscripten_set_main_loop(cb, if (maybe_fps) |fps| fps else -1, @intFromBool(simulate_infinite_loop));
}

//void emscripten_request_animation_frame_loop(EM_BOOL (*cb)(double time, void *userData), void *userData);
pub const AnimationFrameCallback = *const fn (f64, ?*anyopaque) callconv(.C) c_int;
extern fn emscripten_request_animation_frame_loop(AnimationFrameCallback, ?*anyopaque) void;
pub const requestAnimationFrameLoop = emscripten_request_animation_frame_loop;

pub const EmscriptenResult = enum(i16) {
    success = 0,
    deferred = 1,
    not_supported = -1,
    failed_not_deferred = -2,
    invalid_target = -3,
    unknown_target = -4,
    invalid_param = -5,
    failed = -6,
    no_data = -7,
    timed_out = -8,
};
pub const CanvasSizeChangedCallback = *const fn (
    i16,
    *anyopaque,
    ?*anyopaque,
) callconv(.C) c_int;
pub fn setResizeCallback(
    cb: CanvasSizeChangedCallback,
    use_capture: bool,
    user_data: ?*anyopaque,
) EmscriptenResult {
    const result = emscripten_set_resize_callback_on_thread(
        "2",
        user_data,
        @intFromBool(use_capture),
        cb,
        2,
    );
    return @enumFromInt(result);
}
extern fn emscripten_set_resize_callback_on_thread(
    [*:0]const u8,
    ?*anyopaque,
    c_int,
    CanvasSizeChangedCallback,
    c_int,
) c_int;

pub fn getElementCssSize(
    target_id: [:0]const u8,
    width: *f64,
    height: *f64,
) EmscriptenResult {
    return @enumFromInt(emscripten_get_element_css_size(
        target_id,
        width,
        height,
    ));
}
extern fn emscripten_get_element_css_size([*:0]const u8, *f64, *f64) c_int;

/// EmmalocAllocator allocator
/// use with linker flag -sMALLOC=emmalloc
/// for details see docs: https://github.com/emscripten-core/emscripten/blob/main/system/lib/emmalloc.c
pub const EmmalocAllocator = struct {
    const Self = @This();
    dummy: u32 = undefined,

    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = &alloc,
                .resize = &resize,
                .free = &free,
            },
        };
    }

    fn alloc(
        ctx: *anyopaque,
        len: usize,
        ptr_align_log2: u8,
        return_address: usize,
    ) ?[*]u8 {
        _ = ctx;
        _ = return_address;
        const ptr_align: u32 = @as(u32, 1) << @as(u5, @intCast(ptr_align_log2));
        if (!std.math.isPowerOfTwo(ptr_align)) unreachable;
        const ptr = emmalloc_memalign(ptr_align, len) orelse return null;
        return @ptrCast(ptr);
    }

    fn resize(
        ctx: *anyopaque,
        buf: []u8,
        buf_align_log2: u8,
        new_len: usize,
        return_address: usize,
    ) bool {
        _ = ctx;
        _ = return_address;
        _ = buf_align_log2;
        return emmalloc_realloc_try(buf.ptr, new_len) != null;
    }

    fn free(
        ctx: *anyopaque,
        buf: []u8,
        buf_align_log2: u8,
        return_address: usize,
    ) void {
        _ = ctx;
        _ = buf_align_log2;
        _ = return_address;
        return emmalloc_free(buf.ptr);
    }
};

/// std.panic impl
pub fn panic(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = error_return_trace;
    _ = ret_addr;

    var buf: [1024]u8 = undefined;
    const error_msg: [:0]u8 = std.fmt.bufPrintZ(&buf, "PANIC! {s}", .{msg}) catch unreachable;
    emscripten_err(error_msg.ptr);

    while (true) {
        @breakpoint();
    }
}

/// std.log impl
pub fn log(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    const level_txt = comptime level.asText();
    const prefix2 = if (scope == .default) ": " else "(" ++ @tagName(scope) ++ "): ";
    const prefix = level_txt ++ prefix2;

    var buf: [1024]u8 = undefined;
    const msg = std.fmt.bufPrintZ(buf[0 .. buf.len - 1], prefix ++ format, args) catch |err| {
        switch (err) {
            error.NoSpaceLeft => {
                emscripten_console_error("log message too long, skipped.");
                return;
            },
        }
    };
    switch (level) {
        .err => emscripten_console_error(@ptrCast(msg.ptr)),
        .warn => emscripten_console_warn(@ptrCast(msg.ptr)),
        else => emscripten_console_log(@ptrCast(msg.ptr)),
    }
}
