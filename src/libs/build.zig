const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zgpu = b.dependency("zgpu", .{
        .target = target,
    });
    const zmath = b.dependency("zmath", .{
        .target = target,
    });

    _ = b.addModule("shapes", .{
        .root_source_file = b.path("shapes.zig"),
        .imports = &.{
            .{
                .name = "zgpu",
                .module = zgpu.module("root"),
            },
        },
        .target = target,
        .optimize = optimize,
    });

    _ = b.addModule("gui", .{
        .root_source_file = b.path("gui-windows.zig"),
        .imports = &.{
            .{
                .name = "zgpu",
                .module = zgpu.module("root"),
            },
        },
        .target = target,
        .optimize = optimize,
    });

    const camera = b.addModule("camera", .{
        .root_source_file = b.path("camera.zig"),
        .imports = &.{
            .{
                .name = "zgpu",
                .module = zgpu.module("root"),
            },
            .{
                .name = "zmath",
                .module = zmath.module("root"),
            },
        },
        .target = target,
        .optimize = optimize,
    });

    _ = b.addModule("consumer", .{
        .root_source_file = b.path("consumer.zig"),
        .imports = &.{
            .{
                .name = "zgpu",
                .module = zgpu.module("root"),
            },
            .{
                .name = "camera",
                .module = camera,
            },
        },
        .target = target,
        .optimize = optimize,
    });

    _ = b.addModule("producer", .{
        .root_source_file = b.path("producer.zig"),
        .imports = &.{
            .{
                .name = "zgpu",
                .module = zgpu.module("root"),
            },
            .{
                .name = "camera",
                .module = camera,
            },
        },
        .target = target,
        .optimize = optimize,
    });

    const wgpu = b.addModule("wgpu", .{
        .root_source_file = b.path("wgpu.zig"),
        .imports = &.{
            .{
                .name = "zgpu",
                .module = zgpu.module("root"),
            },
            .{
                .name = "zmath",
                .module = zmath.module("root"),
            },
        },
        .target = target,
        .optimize = optimize,
    });

    _ = b.addModule("statistics", .{
        .root_source_file = b.path("statistics.zig"),
        .imports = &.{
            .{
                .name = "zgpu",
                .module = zgpu.module("root"),
            },
            .{
                .name = "wgpu",
                .module = wgpu,
            },
        },
        .target = target,
        .optimize = optimize,
    });
}
