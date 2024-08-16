const std = @import("std");
const Allocator = std.mem.Allocator;
const Timer = std.time.Timer;
const print = std.debug.print;

const zensor = @import("zensor");
const dtypes = zensor.dtypes;
const dtype = dtypes.float32;
const T = dtype.kind;
const Tensor = zensor.Tensor(dtype);

const TRIALS = 100;
const SIZE = 1024;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    try runBenchmark(allocator);
}

fn runBenchmark(allocator: Allocator) !void {
    printBenchmarkInfo();

    var A = try createRandomTensor(allocator, .{ SIZE, SIZE }, 0, 10);
    defer A.deinit();

    var B = try createRandomTensor(allocator, .{ SIZE, SIZE }, 10, 20);
    defer B.deinit();

    var C = try Tensor.full(allocator, .{ SIZE, SIZE }, 0);
    defer C.deinit();

    try validateMatMulAdd(&A, &B, &C);

    const benchmarkResults = try performBenchmark(allocator, &A, &B, &C);
    defer allocator.free(benchmarkResults.times);

    printBenchmarkResults(allocator, benchmarkResults);
}

fn printBenchmarkInfo() void {
    print("Running benchmark:\n", .{});
    print("\tTensor Type: {s}\n", .{@typeName(T)});
    print("\tTensor matmul: two {}x{} Tensors\n", .{ SIZE, SIZE });
    print("\tNumber of Trials: {}\n", .{TRIALS});
}

fn createRandomTensor(allocator: Allocator, shape: anytype, min: T, max: T) !Tensor {
    return switch (@typeInfo(T)) {
        .Float => try Tensor.rand(allocator, shape),
        .Int => try Tensor.randInt(allocator, shape, min, max),
        else => unreachable,
    };
}

fn validateMatMulAdd(A: *const Tensor, B: *const Tensor, C: *Tensor) !void {
    print("Validating...\n", .{});
    _ = try C.matmuladd(A, B);
    if (!validate(A.*, B.*, C.*)) {
        @panic("Validation failed");
    }
}

const BenchmarkResults = struct {
    times: []u64,
    totalTime: u64,
    avgTime: u64,
};

fn performBenchmark(allocator: Allocator, A: *const Tensor, B: *const Tensor, C: *Tensor) !BenchmarkResults {
    print("Starting trials...\n", .{});
    var totalTime: u64 = 0;
    var times = try allocator.alloc(u64, TRIALS);

    for (0..TRIALS) |i| {
        @memset(C.data(), 0);
        const time = try measureMatMulAddTime(A, B, C);
        totalTime += time;
        times[i] = time;

        printProgress(i);
    }
    print("\n", .{});

    return BenchmarkResults{
        .times = times,
        .totalTime = totalTime,
        .avgTime = totalTime / TRIALS,
    };
}

fn measureMatMulAddTime(A: *const Tensor, B: *const Tensor, C: *Tensor) !u64 {
    var timer = try Timer.start();
    _ = try C.matmuladd(A, B);
    return timer.read();
}

fn printProgress(iteration: usize) void {
    const progress = @as(f32, @floatFromInt(iteration + 1)) / @as(f32, @floatFromInt(TRIALS)) * 100;
    print("\rProgress: {}/{} ({d:.1}%)    ", .{ iteration + 1, TRIALS, progress });
}

fn printBenchmarkResults(allocator: Allocator, results: BenchmarkResults) void {
    const avgTimeString = prettyPrintTime(allocator, results.avgTime) catch return;
    defer allocator.free(avgTimeString);

    const stddev = calculateStdDev(@ptrCast(results.times), @as(f64, @floatFromInt(results.avgTime)));
    const stddevString = prettyPrintTime(allocator, @intFromFloat(stddev)) catch return;
    defer allocator.free(stddevString);

    const gflops = calculateGFLOPS(results.avgTime);
    const gflopsStddev = calculateGFLOPSStdDev(results.times, gflops);

    print("Average Time: {s} ± {s}\n", .{ avgTimeString, stddevString });
    print("Performance: {d:.3} ± {d:.3} GFLOPS/s\n", .{ gflops, gflopsStddev });
}

fn prettyPrintTime(allocator: Allocator, duration: u64) ![]const u8 {
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

fn calculateGFLOPSStdDev(times: []const u64, meanGFLOPS: f64) f64 {
    var gflopss: [TRIALS]f64 = .{0} ** TRIALS;
    for (0..TRIALS) |i| {
        gflopss[i] = calculateGFLOPS(times[i]);
    }
    return calculateStdDev(&gflopss, meanGFLOPS);
}

fn calculateStdDev(values: []const f64, mean: f64) f64 {
    var sum_squared_diff: f64 = 0;
    for (values) |value| {
        const diff = value - mean;
        sum_squared_diff += diff * diff;
    }
    return @sqrt(sum_squared_diff / @as(f64, @floatFromInt(values.len)));
}

fn validate(A: Tensor, B: Tensor, C: Tensor) bool {
    for (0..SIZE) |i| {
        for (0..SIZE) |j| {
            var sum: T = 0;
            for (0..SIZE) |k| {
                sum += A.at(@intCast(i * SIZE + k)) * B.at(@intCast(k * SIZE + j));
            }
            switch (@typeInfo(T)) {
                .Float => {
                    const tol = (std.math.floatEps(T) * SIZE) * (1 + std.math.floatEps(T));
                    std.testing.expectApproxEqRel(sum, C.at(@intCast(i * SIZE + j)), tol) catch return false;
                },
                .Int => {
                    std.testing.expectEqual(sum, C.at(@intCast(i * SIZE + j))) catch return false;
                },
                else => unreachable,
            }
        }
    }
    return true;
}
