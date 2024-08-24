const std = @import("std");

const zensor = @import("zensor");
const dtypes = zensor.dtypes;
const LazyTensor = zensor.LazyTensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const T = dtypes.float32;

    var A = try LazyTensor(T).full(allocator, .{ 4, 3 }, 42.0);
    defer A.deinit();

    var B = try LazyTensor(T).full(allocator, .{ 4, 3 }, 6.9);
    defer B.deinit();

    var C = try A.add(&B);
    defer C.deinit();

    try C.evaluate();
    std.debug.print("C: {}\n", .{C.tensor()});
}
