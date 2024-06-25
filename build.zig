const builtin = @import("builtin");
const std = @import("std");

pub const Options = struct {
    optimize: std.builtin.Mode,
    target: std.Build.ResolvedTarget,
};

const demos = struct {
    pub const editor = @import("src/resources/editor/build.zig");
    pub const income = @import("src/resources/income/build.zig");
    pub const random = @import("src/resources/random/build.zig");
    pub const variable = @import("src/resources/variable/build.zig");
};

pub fn build(b: *std.Build) void {
    const options = Options{
        .optimize = b.standardOptimizeOption(.{}),
        .target = b.standardTargetOptions(.{}),
    };

    inline for (comptime std.meta.declarations(demos)) |d| {
        const exe = @field(demos, d.name).build(b, options);

        // TODO: Problems with LTO on Windows.
        if (exe.rootModuleTarget().os.tag == .windows) {
            exe.want_lto = false;
        }

        if (exe.root_module.optimize == .ReleaseFast) {
            exe.root_module.strip = true;
        }

        const install_exe = b.addInstallArtifact(exe, .{});
        b.getInstallStep().dependOn(&install_exe.step);
        b.step(d.name, "Build '" ++ d.name ++ "' demo").dependOn(&install_exe.step);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(&install_exe.step);
        b.step(d.name ++ "-run", "Run '" ++ d.name ++ "' demo").dependOn(&run_cmd.step);
    }
}
