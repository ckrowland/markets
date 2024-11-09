const builtin = @import("builtin");
const std = @import("std");

pub const Options = struct {
    optimize: std.builtin.Mode,
    target: std.Build.ResolvedTarget,
};

pub fn build(b: *std.Build) !void {
    const options = Options{
        .optimize = b.standardOptimizeOption(.{}),
        .target = b.standardTargetOptions(.{}),
    };

    // If emscripten target - library, else - executable
    const emscripten = options.target.result.os.tag == .emscripten;
    const exe_desc = .{
        .name = "Simulations",
        .root_source_file = b.path("src/main.zig"),
        .target = options.target,
        .optimize = options.optimize,
    };
    const exe = blk: {
        if (emscripten) {
            break :blk b.addStaticLibrary(exe_desc);
        } else {
            break :blk b.addExecutable(exe_desc);
        }
    };

    // If user did not set --sysroot then default to emsdk package path
    if (emscripten and b.sysroot == null) {
        b.sysroot = b.dependency("emsdk", .{}).path("upstream/emscripten/cache/sysroot").getPath(b);
        std.log.info("sysroot set to \"{s}\"", .{b.sysroot.?});
    }

    @import("system_sdk").addLibraryPathsTo(exe);
    const zglfw = b.dependency("zglfw", .{
        .target = options.target,
    });
    exe.root_module.addImport("zglfw", zglfw.module("root"));

    @import("zgpu").addLibraryPathsTo(exe);
    const zgpu = b.dependency("zgpu", .{});
    exe.root_module.addImport("zgpu", zgpu.module("root"));

    const zemscripten = b.dependency("zemscripten", .{});
    if (!emscripten) {
        exe.linkLibrary(zglfw.artifact("glfw"));
        exe.linkLibrary(zgpu.artifact("zdawn"));
        exe.root_module.addImport("zemscripten", zemscripten.module("dummy"));
    }

    const zmath = b.dependency("zmath", .{});
    exe.root_module.addImport("zmath", zmath.module("root"));

    const zgui = b.dependency("zgui", .{
        .target = options.target,
        .backend = .glfw_wgpu,
    });
    exe.root_module.addImport("zgui", zgui.module("root"));
    exe.linkLibrary(zgui.artifact("imgui"));

    const zpool = b.dependency("zpool", .{});
    exe.root_module.addImport("zpool", zpool.module("root"));

    const zstbi = b.dependency("zstbi", .{
        .target = options.target,
    });
    exe.root_module.addImport("zstbi", zstbi.module("root"));
    exe.linkLibrary(zstbi.artifact("zstbi"));

    if (emscripten) {
        exe.root_module.addImport("zemscripten", zemscripten.module("root"));
        const zems = @import("zemscripten");
        const activate_emsdk_step = zems.activateEmsdkStep(b);
        const emcc_flags = zems.emccDefaultFlags(b.allocator, options.optimize);
        const emcc_settings = zems.emccDefaultSettings(b.allocator, .{
            .optimize = options.optimize,
        });
        const emcc_step = zems.emccStep(b, exe, .{
            .optimize = options.optimize,
            .flags = emcc_flags,
            .settings = emcc_settings,
            .use_preload_plugins = true,
            .embed_paths = &.{
                .{
                    .src_path = "content",
                    .virtual_path = "/content",
                },
                .{
                    .src_path = "src/random",
                    .virtual_path = "/",
                },
            },
            .preload_paths = &.{},
            .install_dir = .{ .custom = "web" },
            .shell_file_path = "libs/zemscripten/content/shell_minimal.html",
        });
        emcc_step.dependOn(activate_emsdk_step);
        b.getInstallStep().dependOn(emcc_step);

        const html_filename = try std.fmt.allocPrint(b.allocator, "{s}.html", .{exe.name});

        const emrun_args = .{"--no_browser"};
        const emrun_step = zems.emrunStep(
            b,
            b.getInstallPath(.{ .custom = "web" }, html_filename),
            &emrun_args,
        );

        emrun_step.dependOn(emcc_step);

        const run_step = b.step(
            "emrun",
            "Serve and run the web app locally using emrun",
        );
        run_step.dependOn(emrun_step);
    }

    const install_exe = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install_exe.step);
    //b.step("demo", "Build demo").dependOn(&install_exe.step);

    if (!emscripten) {
        const install_content_step = b.addInstallDirectory(.{
            .source_dir = b.path("content"),
            .install_dir = .{ .custom = "" },
            .install_subdir = "bin/content",
        });
        exe.step.dependOn(&install_content_step.step);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(&install_exe.step);
        b.step("run", "Run demo").dependOn(&run_cmd.step);
    }
}
//
//    var release_step = b.step("release", "create executables for all apps");
//    var build_step = b.step("build", "build executables for all apps");
//    inline for (.{
//        .{ .os = .windows, .arch = .x86_64, .output = "apps/Windows/windows-x86_64" },
//        .{ .os = .macos, .arch = .x86_64, .output = "apps/Mac/x86_64/Simulations.app/Contents/MacOS" },
//        .{ .os = .macos, .arch = .aarch64, .output = "apps/Mac/aarch64/Simulations.app/Contents/MacOS" },
//        .{ .os = .linux, .arch = .x86_64, .output = "apps/Linux/linux-x86_64-gnu" },
//        //.{ .os = .linux, .arch = .aarch64, .output = "apps/Linux/aarch64/linux-aarch64" },
//    }) |release| {
//        const target = b.resolveTargetQuery(.{
//            .cpu_arch = release.arch,
//            .os_tag = release.os,
//            .abi = if (release.os == .linux) .gnu else null,
//        });
//        const release_exe = createExe(b, .{
//            .optimize = .ReleaseFast,
//            .target = target,
//        });
//        if (release.os == .macos) {
//            release_exe.headerpad_size = 0x10000;
//        }
//        const install_release = b.addInstallArtifact(release_exe, .{
//            .dest_dir = .{
//                .override = .{ .custom = "../" ++ release.output },
//            },
//        });
//        build_step.dependOn(&install_release.step);
//
//        const install_content_step = b.addInstallDirectory(.{
//            .source_dir = b.path("content"),
//            .install_dir = .{ .custom = "../" ++ release.output },
//            .install_subdir = "content",
//        });
//        build_step.dependOn(&install_content_step.step);
//    }
//
//    const zip_windows = b.addSystemCommand(&.{ "tar", "-cavf" });
//    zip_windows.setCwd(b.path("apps/Windows"));
//    zip_windows.addArg("windows-x86_64.tar.xz");
//    zip_windows.addArg("Windows-x86_64");
//    zip_windows.step.dependOn(build_step);
//
//    const zip_linux = b.addSystemCommand(&.{ "tar", "-cavf" });
//    zip_linux.setCwd(b.path("apps/Linux"));
//    zip_linux.addArg("linux-x86_64-gnu.tar.xz");
//    zip_linux.addArg("linux-x86_64-gnu");
//    zip_linux.step.dependOn(build_step);
//
//    const notarize_apps = b.option(
//        bool,
//        "notarize_apps",
//        "Create an apple signed dmg to distribute",
//    ) orelse false;
//
//    if (notarize_apps) {
//        var notarize_step = b.step("notarize", "notarize macos apps");
//        const dev_id = b.option(
//            []const u8,
//            "developer_id",
//            "Name of certificate used for codesigning macos applications",
//        );
//        const apple_id = b.option(
//            []const u8,
//            "apple_id",
//            "Apple Id used with your Apple Developer account.",
//        );
//        const apple_password = b.option(
//            []const u8,
//            "apple_password",
//            "The Apple app specific password used to notarize applications.",
//        );
//        const apple_team_id = b.option(
//            []const u8,
//            "apple_team_id",
//            "The Apple team ID given on you Apple Developer account.",
//        );
//
//        const x86_64 = notarizeMacApp(
//            b,
//            b.path("apps/Mac/x86_64"),
//            dev_id,
//            apple_id,
//            apple_password,
//            apple_team_id,
//        );
//        const aarch64 = notarizeMacApp(
//            b,
//            b.path("apps/Mac/aarch64"),
//            dev_id,
//            apple_id,
//            apple_password,
//            apple_team_id,
//        );
//        notarize_step.dependOn(x86_64);
//        notarize_step.dependOn(aarch64);
//    }
//
//    release_step.dependOn(build_step);
//    release_step.dependOn(&zip_windows.step);
//    release_step.dependOn(&zip_linux.step);
//}
//
//fn createExe(b: *std.Build, options: Options) *std.Build.Step.Compile {
//    const exe = b.addExecutable(.{
//        .name = "Simulations",
//        .root_source_file = b.path("src/main.zig"),
//        .target = options.target,
//        .optimize = options.optimize,
//    });
//
//    @import("system_sdk").addLibraryPathsTo(exe);
//    const zglfw = b.dependency("zglfw", .{
//        .target = options.target,
//    });
//    exe.root_module.addImport("zglfw", zglfw.module("root"));
//    exe.linkLibrary(zglfw.artifact("glfw"));
//
//    @import("zgpu").addLibraryPathsTo(exe);
//    const zgpu = b.dependency("zgpu", .{
//        .target = options.target,
//    });
//    exe.root_module.addImport("zgpu", zgpu.module("root"));
//    exe.linkLibrary(zgpu.artifact("zdawn"));
//
//    const zmath = b.dependency("zmath", .{
//        .target = options.target,
//    });
//    exe.root_module.addImport("zmath", zmath.module("root"));
//
//    const zgui = b.dependency("zgui", .{
//        .target = options.target,
//        .backend = .glfw_wgpu,
//    });
//    exe.root_module.addImport("zgui", zgui.module("root"));
//    exe.linkLibrary(zgui.artifact("imgui"));
//
//    const zpool = b.dependency("zpool", .{
//        .target = options.target,
//    });
//    exe.root_module.addImport("zpool", zpool.module("root"));
//
//    const zstbi = b.dependency("zstbi", .{
//        .target = options.target,
//    });
//    exe.root_module.addImport("zstbi", zstbi.module("root"));
//    exe.linkLibrary(zstbi.artifact("zstbi"));
//
//    const install_content_step = b.addInstallDirectory(.{
//        .source_dir = b.path("content"),
//        .install_dir = .{ .custom = "" },
//        .install_subdir = "bin/content",
//    });
//    exe.step.dependOn(&install_content_step.step);
//
//    // TODO: Problems with LTO on Windows.
//    if (exe.rootModuleTarget().os.tag == .windows) {
//        exe.want_lto = false;
//    }
//
//    if (exe.root_module.optimize == .ReleaseFast) {
//        exe.root_module.strip = true;
//    }
//
//    return exe;
//}
//
//fn notarizeMacApp(
//    b: *std.Build,
//    path: std.Build.LazyPath,
//    dev_id: ?[]const u8,
//    apple_id: ?[]const u8,
//    apple_password: ?[]const u8,
//    apple_team_id: ?[]const u8,
//) *std.Build.Step {
//    const codesign_app = b.addSystemCommand(&.{
//        "codesign",
//        "-f",
//        "-v",
//        "--deep",
//        "--options=runtime",
//        "--timestamp",
//        "-s",
//        dev_id.?,
//        "Simulations.app",
//    });
//    codesign_app.setCwd(path);
//
//    const create_dmg = b.addSystemCommand(&.{
//        "hdiutil",
//        "create",
//        "-volname",
//        "Simulations",
//        "-srcfolder",
//        "Simulations.app",
//        "-ov",
//        "Simulations.dmg",
//    });
//    create_dmg.setCwd(path);
//    create_dmg.step.dependOn(&codesign_app.step);
//
//    const codesign_dmg = b.addSystemCommand(&.{
//        "codesign",
//        "--timestamp",
//        "-s",
//        dev_id.?,
//        "Simulations.dmg",
//    });
//    codesign_dmg.setCwd(path);
//    codesign_dmg.step.dependOn(&create_dmg.step);
//
//    const notarize_dmg = b.addSystemCommand(&.{
//        "xcrun",
//        "notarytool",
//        "submit",
//        "Simulations.dmg",
//        "--apple-id",
//        apple_id.?,
//        "--password",
//        apple_password.?,
//        "--team-id",
//        apple_team_id.?,
//        "--wait",
//    });
//    notarize_dmg.setCwd(path);
//    notarize_dmg.step.dependOn(&codesign_dmg.step);
//
//    const staple_app = b.addSystemCommand(&.{
//        "xcrun",
//        "stapler",
//        "staple",
//        "Simulations.app",
//    });
//    staple_app.setCwd(path);
//    staple_app.step.dependOn(&notarize_dmg.step);
//
//    const staple_dmg = b.addSystemCommand(&.{
//        "xcrun",
//        "stapler",
//        "staple",
//        "Simulations.dmg",
//    });
//    staple_dmg.setCwd(path);
//    staple_dmg.step.dependOn(&staple_app.step);
//
//    const validate_app = b.addSystemCommand(&.{
//        "xcrun",
//        "stapler",
//        "validate",
//        "Simulations.app",
//    });
//    validate_app.setCwd(path);
//    validate_app.step.dependOn(&staple_dmg.step);
//
//    const validate_dmg = b.addSystemCommand(&.{
//        "xcrun",
//        "stapler",
//        "validate",
//        "Simulations.dmg",
//    });
//    validate_dmg.setCwd(path);
//    validate_dmg.step.dependOn(&validate_app.step);
//    return &validate_dmg.step;
//}
