const std = @import("std");

const T = i32;
const Tensor = @import("zensor").Tensor(T);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var A = try Tensor.full(allocator, .{ 4, 3 }, 4);
    defer A.deinit();
    std.debug.print("{}\n", .{A});

    const slice: []const []const T = &.{
        &.{ -3, -2, -1 },
    };
    var B = try Tensor.fromOwnedSlice(allocator, slice);
    defer B.deinit();
    std.debug.print("{}\n", .{B});

    var C: Tensor = try A.add(&B);
    defer C.deinit();
    std.debug.print("{}\n", .{C});

    var D = try C.mul(2);
    try D.reshape(.{ 2, 6 });
    std.debug.print("{}\n", .{D});
}
