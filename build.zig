const builtin = @import("builtin");
const std = @import("std");

pub const Options = struct {
    optimize: std.builtin.Mode,
    target: std.Build.ResolvedTarget,
};

pub fn build(b: *std.Build) void {
    const options = Options{
        .optimize = b.standardOptimizeOption(.{}),
        .target = b.standardTargetOptions(.{}),
    };

    const exe = createExe(b, options);
    //const native = std.zig.system.NativeTargetInfo.detect(options.target) catch unreachable;
    //std.debug.print("{s}-{s}-{s}\n", .{ @tagName(native.target.cpu.arch), @tagName(native.target.os.tag), @tagName(native.target.abi) });

    const install_exe = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install_exe.step);
    b.step("demo", "Build demo").dependOn(&install_exe.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(&install_exe.step);
    b.step("run", "Run demo").dependOn(&run_cmd.step);

    var release_step = b.step("release", "create executables for all apps");
    const zip_dir_command = b.addSystemCommand(&.{ "zip", "-r9" });
    inline for (.{
        .{ .os = .windows, .arch = .x86_64, .output = "apps/Windows/windows-x86_64" },
        //.{ .os = .macos, .arch = .x86_64, .output = "apps/Mac/x86_64/Simulations.app/Contents/MacOS" },
        .{ .os = .macos, .arch = .aarch64, .output = "apps/Mac/M1/Simulations.app/Contents/MacOS" },
        //.{ .os = .linux, .arch = .x86_64, .output = "apps/Linux/x86_64/linux-x86_64" },
        //.{ .os = .linux, .arch = .aarch64, .output = "apps/Linux/aarch64/linux-aarch64" },
    }) |release| {
        const target = b.resolveTargetQuery(.{
            .cpu_arch = release.arch,
            .os_tag = release.os,
        });
        const release_exe = createExe(b, .{
            .optimize = .ReleaseFast,
            .target = target,
        });
        const install_release = b.addInstallArtifact(release_exe, .{
            .dest_dir = .{
                .override = .{ .custom = "../" ++ release.output },
            },
        });
        release_step.dependOn(&install_release.step);

        if (release.os != .macos) {
            zip_dir_command.addArg(release.output ++ ".zip");
            zip_dir_command.addArg(release.output);
            release_step.dependOn(&zip_dir_command.step);
        }
    }
}

fn createExe(b: *std.Build, options: Options) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "Simulations",
        .root_source_file = b.path("src/main.zig"),
        .target = options.target,
        .optimize = options.optimize,
    });

    @import("system_sdk").addLibraryPathsTo(exe);

    const zglfw = b.dependency("zglfw", .{
        .target = options.target,
    });
    exe.root_module.addImport("zglfw", zglfw.module("root"));
    exe.linkLibrary(zglfw.artifact("glfw"));

    @import("zgpu").addLibraryPathsTo(exe);
    const zgpu = b.dependency("zgpu", .{
        .target = options.target,
    });
    exe.root_module.addImport("zgpu", zgpu.module("root"));
    exe.linkLibrary(zgpu.artifact("zdawn"));

    const zmath = b.dependency("zmath", .{
        .target = options.target,
    });
    exe.root_module.addImport("zmath", zmath.module("root"));

    const zgui = b.dependency("zgui", .{
        .target = options.target,
        .backend = .glfw_wgpu,
    });
    exe.root_module.addImport("zgui", zgui.module("root"));
    exe.linkLibrary(zgui.artifact("imgui"));

    const zpool = b.dependency("zpool", .{
        .target = options.target,
    });
    exe.root_module.addImport("zpool", zpool.module("root"));

    const zstbi = b.dependency("zstbi", .{
        .target = options.target,
    });
    exe.root_module.addImport("zstbi", zstbi.module("root"));
    exe.linkLibrary(zstbi.artifact("zstbi"));

    const install_content_step = b.addInstallDirectory(.{
        .source_dir = b.path("content"),
        .install_dir = .{ .custom = "" },
        .install_subdir = "bin/content",
    });
    exe.step.dependOn(&install_content_step.step);

    // TODO: Problems with LTO on Windows.
    if (exe.rootModuleTarget().os.tag == .windows) {
        exe.want_lto = false;
    }

    if (exe.root_module.optimize == .ReleaseFast) {
        exe.root_module.strip = true;
    }

    return exe;
}
