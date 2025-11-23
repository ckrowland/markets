const builtin = @import("builtin");
const std = @import("std");

const Options = struct {
    optimize: std.builtin.Mode,
    target: std.Build.ResolvedTarget,
};

pub fn build(b: *std.Build) !void {
    const options = Options{
        .optimize = b.standardOptimizeOption(.{}),
        .target = b.standardTargetOptions(.{}),
    };

    const exe = buildNative(b, "src/main.zig", options);

    // TODO: Problems with LTO on Windows.
    if (exe.rootModuleTarget().os.tag == .windows) {
        exe.want_lto = false;
    }

    if (exe.root_module.optimize == .ReleaseFast) {
        exe.root_module.strip = true;
    }

    const install_exe = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install_exe.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(&install_exe.step);
    b.step("run", "Run demo").dependOn(&run_cmd.step);

    var release_step = b.step("release", "create executables for all apps");
    var build_step = b.step("all", "build executables for all apps");
    inline for (.{
        .{ .os = .windows, .arch = .x86_64, .output = "apps/Windows/windows-x86_64" },
        .{ .os = .macos, .arch = .x86_64, .output = "apps/Mac/x86_64/Simulations.app/Contents/MacOS" },
        .{ .os = .macos, .arch = .aarch64, .output = "apps/Mac/aarch64/Simulations.app/Contents/MacOS" },
        .{ .os = .linux, .arch = .x86_64, .output = "apps/Linux/linux-x86_64-gnu" },
    }) |release| {
        const target = b.resolveTargetQuery(.{
            .cpu_arch = release.arch,
            .os_tag = release.os,
            .abi = if (release.os == .linux) .gnu else null,
        });
        const release_exe = buildNative(b, "src/main.zig", .{
            .optimize = .ReleaseFast,
            .target = target,
        });
        if (release.os == .macos) {
            release_exe.headerpad_size = 0x10000;
        }
        const install_release = b.addInstallArtifact(release_exe, .{
            .dest_dir = .{
                .override = .{ .custom = "../" ++ release.output },
            },
        });
        build_step.dependOn(&install_release.step);

        const install_content_step = b.addInstallDirectory(.{
            .source_dir = b.path("content"),
            .install_dir = .{ .custom = "../" ++ release.output },
            .install_subdir = "content",
        });
        build_step.dependOn(&install_content_step.step);
    }

    const zip_windows = b.addSystemCommand(&.{ "tar", "-cavf" });
    zip_windows.setCwd(b.path("apps/Windows"));
    zip_windows.addArg("windows-x86_64.tar.xz");
    zip_windows.addArg("windows-x86_64");
    zip_windows.step.dependOn(build_step);

    const zip_linux = b.addSystemCommand(&.{ "tar", "-cavf" });
    zip_linux.setCwd(b.path("apps/Linux"));
    zip_linux.addArg("linux-x86_64-gnu.tar.xz");
    zip_linux.addArg("linux-x86_64-gnu");
    zip_linux.step.dependOn(build_step);

    const notarize_apps = b.option(
        bool,
        "notarize_apps",
        "Create an apple signed dmg to distribute",
    ) orelse false;

    if (notarize_apps) {
        var notarize_step = b.step("notarize", "notarize macos apps");
        const dev_id = b.option(
            []const u8,
            "developer_id",
            "Name of certificate used for codesigning macos applications",
        );
        const apple_id = b.option(
            []const u8,
            "apple_id",
            "Apple Id used with your Apple Developer account.",
        );
        const apple_password = b.option(
            []const u8,
            "apple_password",
            "The Apple app specific password used to notarize applications.",
        );
        const apple_team_id = b.option(
            []const u8,
            "apple_team_id",
            "The Apple team ID given on you Apple Developer account.",
        );

        const x86_64 = notarizeMacApp(
            b,
            b.path("apps/Mac/x86_64"),
            dev_id,
            apple_id,
            apple_password,
            apple_team_id,
        );
        const aarch64 = notarizeMacApp(
            b,
            b.path("apps/Mac/aarch64"),
            dev_id,
            apple_id,
            apple_password,
            apple_team_id,
        );
        notarize_step.dependOn(x86_64);
        notarize_step.dependOn(aarch64);
    }

    release_step.dependOn(build_step);
    release_step.dependOn(&zip_windows.step);
    release_step.dependOn(&zip_linux.step);
}

pub fn buildNative(
    b: *std.Build,
    comptime root_source_file: [:0]const u8,
    options: anytype,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "exe",
        .root_source_file = b.path(root_source_file),
        .target = options.target,
        .optimize = options.optimize,
    });

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

    const zstbi = b.dependency("zstbi", .{
        .target = options.target,
    });
    exe.root_module.addImport("zstbi", zstbi.module("root"));

    const zpool = b.dependency("zpool", .{});
    exe.root_module.addImport("zpool", zpool.module("root"));

    const zgui = b.dependency("zgui", .{
        .target = options.target,
        .backend = .glfw_wgpu,
        .with_implot = true,
    });
    exe.root_module.addImport("zgui", zgui.module("root"));

    const install_content_step = b.addInstallDirectory(.{
        .source_dir = b.path("content"),
        .install_dir = .{ .custom = "" },
        .install_subdir = "bin/content",
    });
    exe.step.dependOn(&install_content_step.step);

    switch (options.target.result.os.tag) {
        .windows => {
            if (options.target.result.cpu.arch.isX86()) {
                if (options.target.result.abi.isGnu() or options.target.result.abi.isMusl()) {
                    if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
                        exe.addLibraryPath(system_sdk.path("windows/lib/x86_64-windows-gnu"));
                    }
                }
            }
        },
        .macos => {
            if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
                exe.addLibraryPath(system_sdk.path("macos12/usr/lib"));
                exe.addFrameworkPath(system_sdk.path("macos12/System/Library/Frameworks"));
            }
        },
        .linux => {
            if (options.target.result.cpu.arch.isX86()) {
                if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
                    exe.addLibraryPath(system_sdk.path("linux/lib/x86_64-linux-gnu"));
                }
            } else if (options.target.result.cpu.arch == .aarch64) {
                if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
                    exe.addLibraryPath(system_sdk.path("linux/lib/aarch64-linux-gnu"));
                }
            }
        },
        else => {},
    }

    return exe;
}

fn notarizeMacApp(
    b: *std.Build,
    path: std.Build.LazyPath,
    dev_id: ?[]const u8,
    apple_id: ?[]const u8,
    apple_password: ?[]const u8,
    apple_team_id: ?[]const u8,
) *std.Build.Step {
    const codesign_app = b.addSystemCommand(&.{
        "codesign",
        "-f",
        "-v",
        "--deep",
        "--options=runtime",
        "--timestamp",
        "-s",
        dev_id.?,
        "Simulations.app",
    });
    codesign_app.setCwd(path);

    const create_dmg = b.addSystemCommand(&.{
        "hdiutil",
        "create",
        "-volname",
        "Simulations",
        "-srcfolder",
        "Simulations.app",
        "-ov",
        "Simulations.dmg",
    });
    create_dmg.setCwd(path);
    create_dmg.step.dependOn(&codesign_app.step);

    const codesign_dmg = b.addSystemCommand(&.{
        "codesign",
        "--timestamp",
        "-s",
        dev_id.?,
        "Simulations.dmg",
    });
    codesign_dmg.setCwd(path);
    codesign_dmg.step.dependOn(&create_dmg.step);

    const notarize_dmg = b.addSystemCommand(&.{
        "xcrun",
        "notarytool",
        "submit",
        "Simulations.dmg",
        "--apple-id",
        apple_id.?,
        "--password",
        apple_password.?,
        "--team-id",
        apple_team_id.?,
        "--wait",
    });
    notarize_dmg.setCwd(path);
    notarize_dmg.step.dependOn(&codesign_dmg.step);

    const staple_app = b.addSystemCommand(&.{
        "xcrun",
        "stapler",
        "staple",
        "Simulations.app",
    });
    staple_app.setCwd(path);
    staple_app.step.dependOn(&notarize_dmg.step);

    const staple_dmg = b.addSystemCommand(&.{
        "xcrun",
        "stapler",
        "staple",
        "Simulations.dmg",
    });
    staple_dmg.setCwd(path);
    staple_dmg.step.dependOn(&staple_app.step);

    const validate_app = b.addSystemCommand(&.{
        "xcrun",
        "stapler",
        "validate",
        "Simulations.app",
    });
    validate_app.setCwd(path);
    validate_app.step.dependOn(&staple_dmg.step);

    const validate_dmg = b.addSystemCommand(&.{
        "xcrun",
        "stapler",
        "validate",
        "Simulations.dmg",
    });
    validate_dmg.setCwd(path);
    validate_dmg.step.dependOn(&validate_app.step);
    return &validate_dmg.step;
}
