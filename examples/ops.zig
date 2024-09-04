const std = @import("std");

const zensor = @import("zensor");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // ** Load in Numpy tensor as u8 buffer **
    const filename: []const u8 = "./examples/numpy.npy";
    var buffer: *zensor.RuntimeBuffer = blk: {
        var buffer = try zensor.RuntimeBuffer.Numpy.load(allocator, filename);
        break :blk &buffer;
    };
    defer buffer.deinit();

    // ** create comptime nodes **
    const dtype = comptime zensor.dtypes.int64;

    const load_node = comptime blk: {
        const shape = &[_]u32{3};
        const view = zensor.View(shape).init().as_any_view();
        const node = zensor.ast.Node.init(
            .Load,
            .{ .name = filename },
            {},
            view,
            dtype,
        );

        break :blk &node;
    };

    const const_node = comptime blk: {
        const shape = &[_]u32{3};
        const view = zensor.View(shape).init().as_any_view();
        const node = zensor.ast.Node.init(
            .Const,
            .{ .value = std.fmt.comptimePrint("{}", .{4}) },
            {},
            view,
            dtype,
        );
        break :blk &node;
    };

    const mul_node = comptime blk: {
        const shape = &[_]u32{3};
        const view = zensor.View(shape).init().as_any_view();
        const node = zensor.ast.Node.init(
            .Mul,
            {},
            [_]*const zensor.ast.Node{ load_node, const_node },
            view,
            dtype,
        );
        break :blk &node;
    };

    const sum_node = comptime blk: {
        const shape = &[_]u32{1};
        const view = zensor.View(shape).init().as_any_view();
        const node = zensor.ast.Node.init(
            .Sum,
            .{ .dim = 0 },
            [_]*const zensor.ast.Node{mul_node},
            view,
            dtype,
        );
        break :blk &node;
    };

    // ** schedule nodes **
    var scheduler = zensor.Scheduler.init(allocator);
    defer scheduler.deinit();
    try scheduler.register_buffer(load_node, buffer);
    try scheduler.mark_for_scheduling(sum_node);

    const schedule = try scheduler.run(sum_node);
    std.debug.print("{}\n", .{schedule});

    // ** IR generator **
    var ir_generator = zensor.IRGenerator.init(allocator);

    const ir_block = try ir_generator.run(schedule);
    std.debug.print("{}\n", .{ir_block});
}
