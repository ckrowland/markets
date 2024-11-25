const std = @import("std");
const variable = @import("main.zig");

pub fn main() !void {
    { // Change current working directory to where the executable is located.
        var buffer: [1024]u8 = undefined;
        const path = std.fs.selfExeDirPath(buffer[0..]) catch ".";
        try std.posix.chdir(path);
    }

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var demo = try variable.init(allocator);
    defer variable.deinit(&demo);

    while (demo.window.shouldClose() == false) {
        try variable.updateAndRender(&demo);
    }
}
