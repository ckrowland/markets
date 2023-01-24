const builtin = @import("builtin");
const std = @import("std");
const zglfw = @import("libs/zglfw/build.zig");
const zgpu = @import("libs/zgpu/build.zig");
const zgui = @import("libs/zgui/build.zig");
const zmath = @import("libs/zmath/build.zig");
const zpool = @import("libs/zpool/build.zig");
const content_dir = "content/";

pub fn build(b: *std.build.Builder) void {
    var options = Options{
        .build_mode = b.standardReleaseOptions(),
        .target = b.standardTargetOptions(.{}),
    };

    options.ztracy_enable = b.option(bool, "ztracy-enable", "Enable Tracy profiler") orelse false;
    options.zgpu_dawn_from_source = b.option(
        bool,
        "zgpu-dawn-from-source",
        "Build Dawn (wgpu implementation) from source",
    ) orelse false;

    if (options.zgpu_dawn_from_source) {
        ensureSubmodules(b.allocator) catch |err| @panic(@errorName(err));
    }

    const install = b.step("demos", "Build demos");

    var exe = createExe(b, options);
    install.dependOn(&b.addInstallArtifact(exe).step);

    const run_step = b.step("demos-run", "Run demos");
    const run_cmd = exe.run();
    run_cmd.step.dependOn(install);
    run_step.dependOn(&run_cmd.step);

    b.getInstallStep().dependOn(install);
}

pub const Options = struct {
    build_mode: std.builtin.Mode,
    target: std.zig.CrossTarget,

    ztracy_enable: bool = false,
    zpix_enable: bool = false,
    zgpu_dawn_from_source: bool = false,

    enable_dx_debug: bool = false,
    enable_dx_gpu_debug: bool = false,
};

fn installDemo(b: *std.build.Builder, exe: *std.build.LibExeObjStep, comptime name: []const u8) void {
    comptime var desc_name: [256]u8 = [_]u8{0} ** 256;
    comptime _ = std.mem.replace(u8, name, "_", " ", desc_name[0..]);
    comptime var desc_size = std.mem.indexOf(u8, &desc_name, "\x00").?;

    const install = b.step(name, "Build '" ++ desc_name[0..desc_size] ++ "' demo");
    install.dependOn(&b.addInstallArtifact(exe).step);

    const run_step = b.step(name ++ "-run", "Run '" ++ desc_name[0..desc_size] ++ "' demo");
    const run_cmd = exe.run();
    run_cmd.step.dependOn(install);
    run_step.dependOn(&run_cmd.step);

    b.getInstallStep().dependOn(install);
}

fn ensureSubmodules(allocator: std.mem.Allocator) !void {
    if (std.process.getEnvVarOwned(allocator, "NO_ENSURE_SUBMODULES")) |no_ensure_submodules| {
        if (std.mem.eql(u8, no_ensure_submodules, "true")) return;
    } else |_| {}
    var child = std.ChildProcess.init(&.{ "git", "submodule", "update", "--init", "--recursive" }, allocator);
    child.cwd = thisDir();
    child.stderr = std.io.getStdErr();
    child.stdout = std.io.getStdOut();
    _ = try child.spawnAndWait();
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}

fn createExe(b: *std.build.Builder, options: Options) *std.build.LibExeObjStep {
    const exe = b.addExecutable("Visual Simulations", thisDir() ++ "/src/main.zig");

    const exe_options = b.addOptions();
    exe.addOptions("build_options", exe_options);
    exe_options.addOption([]const u8, "content_dir", thisDir() ++ "/" ++ content_dir);

    const install_content_step = b.addInstallDirectory(.{
        .source_dir = thisDir() ++ "/" ++ content_dir,
        .install_dir = .{ .custom = "" },
        .install_subdir = "bin/" ++ content_dir,
    });
    exe.step.dependOn(&install_content_step.step);

    exe.setBuildMode(options.build_mode);
    exe.setTarget(options.target);

    const zgui_options = zgui.BuildOptionsStep.init(b, .{ .backend = .glfw_wgpu });
    const zgui_pkg = zgui.getPkg(&.{zgui_options.getPkg()});

    const zgpu_options = zgpu.BuildOptionsStep.init(b, .{});
    const zgpu_pkg = zgpu.getPkg(&.{ zgpu_options.getPkg(), zpool.pkg, zglfw.pkg });

    exe.addPackage(zgpu_pkg);
    exe.addPackage(zmath.pkg);
    exe.addPackage(zglfw.pkg);
    exe.addPackage(zgui_pkg);

    zgpu.link(exe, zgpu_options);
    zglfw.link(exe);
    zgui.link(exe, zgui_options);

    return exe;
}
