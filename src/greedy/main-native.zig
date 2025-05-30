const std = @import("std");
const random = @import("main.zig");

pub fn main() !void {
    { // Change current working directory to where the executable is located.
        var buffer: [1024]u8 = undefined;
        const path = std.fs.selfExeDirPath(buffer[0..]) catch ".";
        try std.posix.chdir(path);
    }

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var demo = try random.init(allocator);
    defer random.deinit(&demo);

    while (demo.window.shouldClose() == false) {
        try random.updateAndRender(&demo);
    }
}
