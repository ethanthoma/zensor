const std = @import("std");

const zensor = @import("zensor");
const Scheduler = zensor.Scheduler;
const Tensor = zensor.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var scheduler: Scheduler = zensor.Scheduler.init(allocator);

    const dtype = zensor.dtypes.int64;
    const shape: []const u32 = &[_]u32{3};
    const filename = "./examples/numpy.npy";

    const a = try Tensor(dtype, shape).from_numpy(&scheduler, filename);

    const b = try Tensor(dtype, shape).full(&scheduler, 4);

    const c = try a.mul(b);

    const d = try c.sum(0);

    d.realize();
}
