const std = @import("std");

const zensor = @import("zensor");
const dtypes = zensor.dtypes;
const Tensor = @import("zensor").Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const dtype = dtypes.bfloat16;

    var A = try Tensor(dtype).full(allocator, .{ 4, 3 }, 42.0);
    defer A.deinit();
    std.debug.print("{}\n", .{A});

    var B = try Tensor(dtype).full(allocator, .{ 4, 3 }, 6.9);
    defer B.deinit();
    std.debug.print("{}\n", .{B});

    var C = try A.add(&B);
    defer C.deinit();
    std.debug.print("{}\n", .{C});
}
