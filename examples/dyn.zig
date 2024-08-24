const std = @import("std");

const zensor = @import("zensor");
const Tensor = zensor.DynamicTensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const A = try Tensor.full(allocator, .Float32, 14, 4);
    defer A.deinit();

    A.printData();
}
