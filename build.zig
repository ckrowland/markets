const builtin = @import("builtin");
const std = @import("std");
const zglfw = @import("libs/zglfw/build.zig");
const zgpu = @import("libs/zgpu/build.zig");
const zgui = @import("libs/zgui/build.zig");
const zmath = @import("libs/zmath/build.zig");
const zpool = @import("libs/zpool/build.zig");
const zstbi = @import("libs/zstbi/build.zig");
var content_dir: []const u8 = "./content/";

pub const Options = struct {
    optimize: std.builtin.Mode,
    target: std.zig.CrossTarget,
};

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    if (target.os_tag == std.Target.Os.Tag.macos) {
        content_dir = "../Resources/";
    }

    var options = Options{
        .optimize = b.standardOptimizeOption(.{}),
        .target = target,
    };

    const install_step = b.step("demos", "Build demos");

    var exe = createExe(b, options);
    install_step.dependOn(&b.addInstallArtifact(exe).step);
    
    const run_step = b.step("demos-run", "Run demos");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(install_step);
    run_step.dependOn(&run_cmd.step);

    b.getInstallStep().dependOn(install_step);
}

fn createExe(b: *std.build.Builder, options: Options) *std.build.LibExeObjStep {
    const target = options.target;
    const optimize = options.optimize;
    
    const exe = b.addExecutable(.{
        .name = "Visual Simulations",
        .root_source_file = .{ .path = thisDir() ++ "/src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    const zgui_pkg = zgui.package(b, target, optimize, .{
        .options = .{ .backend = .glfw_wgpu },
    });
    zgui_pkg.link(exe);

    const zglfw_pkg = zglfw.package(b, target, optimize, .{});
    const zpool_pkg = zpool.package(b, target, optimize, .{});
    const zgpu_pkg = zgpu.package(b, target, optimize, .{
        .deps = .{ .zpool = zpool_pkg.zpool, .zglfw = zglfw_pkg.zglfw },
    });
    const zmath_pkg = zmath.package(b, target, optimize, .{
        .options = .{ .enable_cross_platform_determinism = true },
    });
    const zstbi_pkg = zstbi.package(b, target, optimize, .{});
    
    zglfw_pkg.link(exe);
    zgpu_pkg.link(exe);
    zmath_pkg.link(exe);
    zstbi_pkg.link(exe);
    
    const exe_options = b.addOptions();
    exe.addOptions("build_options", exe_options);
    exe_options.addOption([]const u8, "content_dir", content_dir);
    
    const install_content_step = b.addInstallDirectory(.{
        .source_dir = .{ .path = thisDir() ++ "/content/" },
        .install_dir = .{ .custom = "" },
        .install_subdir = "bin/content/",
    });
    exe.step.dependOn(&install_content_step.step);

    return exe;
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
