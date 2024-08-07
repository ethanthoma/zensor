const std = @import("std");
const Allocator = std.mem.Allocator;
const Timter = std.time.Timer;
const print = std.debug.print;

const T = f16;
const Tensor = @import("zensor").Tensor(T);

const TRIALS = 1000;
const SIZE = 128;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    try trial(allocator);
}

fn trial(allocator: Allocator) !void {
    print("Starting trials...\n", .{});
    var A = try Tensor.empty(allocator, .{ SIZE, SIZE });
    defer A.deinit();

    var B = try Tensor.empty(allocator, .{ SIZE, SIZE });
    defer B.deinit();

    var totalTime: u64 = 0;
    const progressInterval = @divFloor(TRIALS, 10);
    for (0..TRIALS) |i| {
        const step = struct {
            inline fn step(_A: *const Tensor, _B: *const Tensor) !u64 {
                var a = _A.*;
                var b = _B.*;
                var timer = try std.time.Timer.start();
                var C: Tensor = try a.matmul(&b);
                defer C.deinit();
                return timer.read();
            }
        }.step;
        totalTime += try step(&A, &B);

        if (@mod((i + 1), progressInterval) == 0 or i == TRIALS - 1) {
            const progress = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(TRIALS)) * 100;
            print("\rProgress: {}/{} ({d:.1}%)    ", .{ i + 1, TRIALS, progress });
        }
    }

    const timeAsString = try prettyPrintTime(allocator, totalTime);
    defer allocator.free(timeAsString);

    const gflops = calculateGFLOPS(totalTime / TRIALS);

    print("\nMatrix multiplication of two {}x{} matrices:\n", .{ SIZE, SIZE });
    print("Total time: {s} for {} trials.\n", .{ timeAsString, TRIALS });
    print("Performance: {d:.2} GFLOPS\n", .{gflops});
}

fn prettyPrintTime(allocator: std.mem.Allocator, duration: u64) ![]const u8 {
    const sec = @as(f64, @floatFromInt(duration)) / 1_000_000_000;
    const ms = @as(f64, @floatFromInt(duration)) / 1_000_000;
    const us = @as(f64, @floatFromInt(duration)) / 1_000;

    if (sec >= 1) {
        return try std.fmt.allocPrint(allocator, "{:.3} s", .{sec});
    } else if (ms >= 1) {
        return try std.fmt.allocPrint(allocator, "{:.3} ms", .{ms});
    } else if (us >= 1) {
        return try std.fmt.allocPrint(allocator, "{:.3} us", .{us});
    } else {
        return try std.fmt.allocPrint(allocator, "{} ns", .{duration});
    }
}

fn calculateGFLOPS(durationNs: u64) f64 {
    const operations = 2 * SIZE * SIZE * SIZE;
    const durationSec = @as(f64, @floatFromInt(durationNs)) / 1_000_000_000;
    return @as(f64, @floatFromInt(operations)) / durationSec / 1_000_000_000;
}
