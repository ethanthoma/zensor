const std = @import("std");

const zensor = @import("zensor");

pub const std_options = std.Options{ .log_level = .info };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var compiler = zensor.Compiler.init(allocator);
    defer compiler.deinit();

    const filename = "./examples/numpy.npy";

    const a = try zensor.Tensor(.Int64, .{3}).from_numpy(&compiler, filename);

    const b = try zensor.Tensor(.Int64, .{3}).full(&compiler, 4);

    const c = try a.mul(b);

    const d = try c.sum(0);

    std.debug.print("{}\n", .{c});
    std.debug.print("{}\n", .{d});
    std.debug.print("{}\n", .{d});
}
