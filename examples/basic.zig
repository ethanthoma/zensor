const std = @import("std");

const zensor = @import("zensor");
const compiler = zensor.compiler;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var scheduler = compiler.Scheduler.init(allocator);

    const filename = "./examples/numpy.npy";

    const a = try zensor.Tensor(.Int64, .{3}).from_numpy(&scheduler, filename);

    const b = try zensor.Tensor(.Int64, .{3}).full(&scheduler, 4);

    const c = try a.mul(b);

    const d = try c.sum(0);

    std.debug.print("{}\n", .{d});
}
