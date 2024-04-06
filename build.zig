const builtin = @import("builtin");
const std = @import("std");
const zems = @import("libs/zems/build.zig");
const zglfw = @import("libs/zglfw/build.zig");
const zgpu = @import("libs/zgpu/build.zig");
const zgui = @import("libs/zgui/build.zig");
const zmath = @import("libs/zmath/build.zig");
const zpool = @import("libs/zpool/build.zig");
const zstbi = @import("libs/zstbi/build.zig");
const editor = @import("src/resources/editor/build.zig");
const random = @import("src/resources/random/build.zig");
const income = @import("src/resources/income/build.zig");
pub var zems_pkg: zems.Package = undefined;
pub var zglfw_pkg: zglfw.Package = undefined;
pub var zgpu_pkg: zgpu.Package = undefined;
pub var zgui_glfw_wgpu_pkg: zgui.Package = undefined;
pub var zmath_pkg: zmath.Package = undefined;
pub var zpool_pkg: zpool.Package = undefined;
pub var zstbi_pkg: zstbi.Package = undefined;
var content_dir: []const u8 = "./content/";

pub const Options = struct {
    optimize: std.builtin.Mode,
    target: std.zig.CrossTarget,
};

pub fn build(b: *std.build.Builder) void {
    var options = Options{
        .optimize = b.standardOptimizeOption(.{}),
        .target = b.standardTargetOptions(.{}),
    };

    packagesCrossPlatform(b, options);

    if (options.target.getOsTag() == .emscripten) {
        const editor_step = buildEmscripten(b, options, editor.build(b, options));
        var install_step = b.step("editor-web", "Build editor webpage");
        install_step.dependOn(&editor_step.link_step.?.step);
        b.getInstallStep().dependOn(install_step);

        const random_step = buildEmscripten(b, options, random.build(b, options));
        install_step = b.step("random-web", "Build random webpage");
        install_step.dependOn(&random_step.link_step.?.step);
        b.getInstallStep().dependOn(install_step);

        const income_step = buildEmscripten(b, options, income.build(b, options));
        install_step = b.step("income-web", "Build income webpage");
        install_step.dependOn(&income_step.link_step.?.step);
        b.getInstallStep().dependOn(install_step);
    } else {
        install(b, editor.build(b, options), "editor");
        install(b, random.build(b, options), "random");
        install(b, income.build(b, options), "income");
    }
}

pub fn buildEmscripten(
    b: *std.Build,
    options: Options,
    exe: *std.Build.CompileStep,
) *zems.EmscriptenStep {
    var ems_step = zems.EmscriptenStep.init(b);
    ems_step.args.setDefault(options.optimize, false);
    ems_step.args.setOrAssertOption("USE_GLFW", "3");
    ems_step.args.setOrAssertOption("USE_WEBGPU", "");
    ems_step.args.other_args.appendSlice(
        &.{ "--preload-file", "content" },
    ) catch unreachable;
    ems_step.link(exe);
    return ems_step;
}

fn install(
    b: *std.Build,
    exe: *std.Build.CompileStep,
    comptime name: []const u8,
) void {
    const install_step = b.step(name, "Build '" ++ name ++ "' demo");
    install_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    const run_step = b.step(name ++ "-run", "Run '" ++ name ++ "' demo");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(install_step);
    run_step.dependOn(&run_cmd.step);

    b.getInstallStep().dependOn(install_step);
}

fn packagesCrossPlatform(b: *std.Build, options: Options) void {
    const target = options.target;
    const optimize = options.optimize;

    zmath_pkg = zmath.package(b, target, optimize, .{});
    zpool_pkg = zpool.package(b, target, optimize, .{});
    zglfw_pkg = zglfw.package(b, target, optimize, .{});
    zstbi_pkg = zstbi.package(b, target, optimize, .{});
    zgui_glfw_wgpu_pkg = zgui.package(b, target, optimize, .{
        .options = .{ .backend = .glfw_wgpu },
    });
    zgpu_pkg = zgpu.package(b, target, optimize, .{
        .options = .{ .uniforms_buffer_size = 4 * 1024 * 1024 },
        .deps = .{ .zpool = zpool_pkg.zpool, .zglfw = zglfw_pkg.zglfw },
    });
    zems_pkg = zems.package(b, target, optimize, .{});
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
