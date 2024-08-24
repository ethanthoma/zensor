const std = @import("std");

const zensor = @import("zensor");
const dtype = zensor.dtypes.float32;
const Tensor = zensor.Tensor(dtype);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var A = try Tensor.load(allocator, "examples/model.safetensor", .Safetensor);
    defer A.deinit();
    std.debug.print("{}\n", .{A});
}
