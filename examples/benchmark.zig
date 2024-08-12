const std = @import("std");
const Allocator = std.mem.Allocator;
const Timter = std.time.Timer;
const debug = std.debug;
const assert = debug.assert;
const print = debug.print;

const zensor = @import("zensor");
const dtypes = zensor.dtypes;
const dtype = dtypes.float32;
const T = dtype.kind;
const Tensor = @import("zensor").Tensor;

const TRIALS = 1000;
const SIZE = 512;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    try trial(allocator);
}

fn trial(allocator: Allocator) !void {
    print("Running benchmark:\n", .{});
    print("\tTensor Type: {s}\n", .{@typeName(T)});
    print("\tTensor matmul: two {}x{} Tensors\n", .{ SIZE, SIZE });
    print("\tNumber of Trials: {}\n", .{TRIALS});

    var A = switch (@typeInfo(T)) {
        .Float => try Tensor.rand(allocator, .{ SIZE, SIZE }, dtype),
        .Int => try Tensor.randInt(allocator, .{ SIZE, SIZE }, 0, 10, dtype),
        else => unreachable,
    };
    defer A.deinit();

    var B = switch (@typeInfo(T)) {
        .Float => try Tensor.rand(allocator, .{ SIZE, SIZE }),
        .Int => try Tensor.randInt(allocator, .{ SIZE, SIZE }, 10, 20),
        else => unreachable,
    };
    defer B.deinit();

    var C = try Tensor.empty(allocator, .{ SIZE, SIZE });
    defer C.deinit();

    @memset(C.data(), 0);
    print("Validating...\n", .{});
    _ = try C.matmuladd(&A, &B);
    assert(validate(A, B, C));

    print("Starting trials...\n", .{});
    var totalTime: u64 = 0;
    for (0..TRIALS) |i| {
        @memset(C.data(), 0);
        const step = struct {
            inline fn step(_A: *const Tensor, _B: *const Tensor, _C: *const Tensor) !u64 {
                var a = _A.*;
                var b = _B.*;
                var c = _C.*;
                var timer = try std.time.Timer.start();
                _ = try c.matmuladd(&a, &b);
                return timer.read();
            }
        }.step;
        totalTime += try step(&A, &B, &C);

        const progress = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(TRIALS)) * 100;
        print("\rProgress: {}/{} ({d:.1}%)    ", .{ i + 1, TRIALS, progress });
    }
    print("\n", .{});

    const timeAsString = try prettyPrintTime(allocator, totalTime);
    defer allocator.free(timeAsString);

    const gflops = calculateGFLOPS(totalTime / TRIALS);

    print("Total time: {s} for {} trials.\n", .{ timeAsString, TRIALS });
    print("Performance: {d:.3} GFLOPS/s\n", .{gflops});
}

fn prettyPrintTime(allocator: std.mem.Allocator, duration: u64) ![]const u8 {
    const sec = @as(f64, @floatFromInt(duration)) / 1_000_000_000;
    const ms = @as(f64, @floatFromInt(duration)) / 1_000_000;
    const us = @as(f64, @floatFromInt(duration)) / 1_000;

    if (sec >= 1) {
        return try std.fmt.allocPrint(allocator, "{d:.3} s", .{sec});
    } else if (ms >= 1) {
        return try std.fmt.allocPrint(allocator, "{d:.3} ms", .{ms});
    } else if (us >= 1) {
        return try std.fmt.allocPrint(allocator, "{d:.3} us", .{us});
    } else {
        return try std.fmt.allocPrint(allocator, "{d} ns", .{duration});
    }
}

fn calculateGFLOPS(durationNs: u64) f64 {
    const operations = 2 * SIZE * SIZE * SIZE;
    const durationSec = @as(f64, @floatFromInt(durationNs)) / 1_000_000_000;
    return @as(f64, @floatFromInt(operations)) / durationSec / 1_000_000_000;
}

fn validate(A: Tensor, B: Tensor, C: Tensor) bool {
    for (0..SIZE) |i| {
        for (0..SIZE) |j| {
            var sum: T = 0;
            for (0..SIZE) |k| {
                sum += A.at(i * SIZE + k) * B.at(k * SIZE + j);
            }
            switch (@typeInfo(T)) {
                .Float => {
                    const tol = (std.math.floatEps(T) * SIZE) * (1 + std.math.floatEps(T));
                    std.testing.expectApproxEqRel(sum, C.at(i * SIZE + j), tol) catch return false;
                },
                .Int => {
                    std.testing.expectEqual(sum, C.at(i * SIZE + j)) catch return false;
                },
                else => unreachable,
            }
        }
    }
    return true;
}
