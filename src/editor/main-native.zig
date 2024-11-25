const std = @import("std");
const editor = @import("main.zig");

pub fn main() !void {
    { // Change current working directory to where the executable is located.
        var buffer: [1024]u8 = undefined;
        const path = std.fs.selfExeDirPath(buffer[0..]) catch ".";
        try std.posix.chdir(path);
    }

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var demo = try editor.init(allocator);
    defer editor.deinit(&demo);

    while (demo.window.shouldClose() == false) {
        try editor.updateAndRender(&demo);
    }
}
