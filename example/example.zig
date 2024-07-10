const std = @import("std");

const Tensor = @import("zensor").Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var A = try Tensor(u64).init(allocator, ([_]usize{ 2, 3 })[0..]);
    defer A.deinit();
    A.fill(4);
    std.debug.print("{}\n", .{A});

    const slice: []const []const u64 = &.{ &.{ 3, 2, 1 }, &.{ 3, 2, 1 } };
    var B = try Tensor(u64).fromOwnedSlice(allocator, slice);
    defer B.deinit();
    std.debug.print("{}\n", .{B});

    const C = try A.add(&B);
    defer C.deinit();
    std.debug.print("{}\n", .{C});

    const D = try Tensor(u32).arange(allocator, 5, 12);
    defer D.deinit();
    std.debug.print("{}\n", .{D});
}
