const builtin = @import("builtin");
const std = @import("std");

// (Note to myself, users don't need to do this)
// To update submodules do:
//
// 1. Edit .gitmodules (update Dawn branch)
// 2. git submodule update --remote --recursive
// 3. git add .
// 4. git commit -m "update submodules"

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

    //
    // Cross-platform demos
    //
    if (!builtin.is_test) {
        installDemo(b, resources.build(b, options), "resources");
        installDemo(b, bloodstream.build(b, options), "bloodstream");
        installDemo(b, editor.build(b, options), "editor");
    }

    //
    // Tests
    //
    const test_step = b.step("test", "Run all tests");

    const zpool_tests = @import("libs/zpool/build.zig").buildTests(b, options.build_mode, options.target);
    test_step.dependOn(&zpool_tests.step);
    const zgpu_tests = @import("libs/zgpu/build.zig").buildTests(b, options.build_mode, options.target);
    test_step.dependOn(&zgpu_tests.step);
    const zmath_tests = zmath.buildTests(b, options.build_mode, options.target);
    test_step.dependOn(&zmath_tests.step);

    //
    // Benchmarks
    //
    if (!builtin.is_test) {
        const benchmark_step = b.step("benchmark", "Run all benchmarks");
        {
            const run_cmd = zmath.buildBenchmarks(b, options.target).run();
            benchmark_step.dependOn(&run_cmd.step);
        }
    }
}

const zmath = @import("libs/zmath/build.zig");

const resources = @import("src/resources/build.zig");
const bloodstream = @import("src/bloodstream/build.zig");
const editor = @import("src/editor/build.zig");


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
