const std = @import("std");

const zensor = @import("zensor");
const dtypes = zensor.dtypes;
const Tensor = zensor.Tensor;
const Node = zensor.Node;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const T = dtypes.float16;
    const S = dtypes.float32;

    var A = try Tensor(T).full(allocator, .{ 4, 3 }, 42.0);
    defer A.deinit();

    var node_A = try Node(T, T).constant(allocator, &A);
    defer node_A.deinit();

    const opq = node_A.toOpaqueNode();

    var node_A_cast = try Node(T, S).cast(allocator, opq);
    defer node_A_cast.deinit();

    // you can evaluate wherever
    const val_A = try node_A_cast.evaluate();
    _ = val_A;

    var B = try Tensor(S).full(allocator, .{ 4, 3 }, 6.9);
    defer B.deinit();

    var node_B = try Node(S, S).constant(allocator, &B);
    defer node_B.deinit();

    var node_C = try Node(S, S).add(allocator, node_A_cast.toOpaqueNode(), node_B.toOpaqueNode());
    defer node_C.deinit();

    // this wont re-evaluate node_A_cast but use its value instead
    const val_C = try node_C.evaluate();
    std.debug.print("node_C: {}\n", .{val_C});
}
