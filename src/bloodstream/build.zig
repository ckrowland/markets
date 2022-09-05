const std = @import("std");
const zgpu = @import("../../zig-gamedev/libs/zgpu/build.zig");
const zmath = @import("../../zig-gamedev/libs/zmath/build.zig");
const zpool = @import("../../zig-gamedev/libs/zpool/build.zig");

const Options = @import("../../build.zig").Options;
const content_dir = "content/";

pub fn build(b: *std.build.Builder, options: Options) *std.build.LibExeObjStep {
    const exe = b.addExecutable("Circulatory System", thisDir() ++ "/src/bloodstream.zig");

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

    const zgpu_options = zgpu.BuildOptionsStep.init(b, .{
        .dawn = .{},
    });
    const zgpu_pkg = zgpu.getPkg(&.{ zgpu_options.getPkg(), zpool.pkg });

    exe.addPackage(zgpu_pkg);
    exe.addPackage(zmath.pkg);

    zgpu.link(exe, zgpu_options);

    return exe;
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
