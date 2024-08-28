const std = @import("std");

const zensor = @import("zensor");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // ** Load in Numpy tensor as u8 buffer **
    const n = try zensor.buffer.Numpy.load(allocator, "./examples/numpy.npy");

    // ** create comptime buffer id **
    const buffer_id = comptime zensor.MemoryManager.create_id(opaque {});

    // ** create memory manager that links comptime nodes with runtime buffers **
    var memory_manager = zensor.MemoryManager.init();
    try memory_manager.set_buffer(allocator, buffer_id, n);

    // ** create comptime nodes **
    const dtype = comptime zensor.dtypes.int64;

    const load_node = comptime blk: {
        const shape = &[_]u32{3};
        const view = zensor.View(shape).init().to_any_view();
        const node = zensor.ast.Node.init(
            .Load,
            .{ .buffer_id = buffer_id },
            {},
            view,
            dtype,
        );

        break :blk &node;
    };

    const const_node = comptime blk: {
        const shape = &[_]u32{3};
        const view = zensor.View(shape).init().to_any_view();
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
        const view = zensor.View(shape).init().to_any_view();
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
        const view = zensor.View(shape).init().to_any_view();
        const node = zensor.ast.Node.init(
            .Sum,
            .{ .dim = 0 },
            [_]*const zensor.ast.Node{mul_node},
            view,
            dtype,
        );
        break :blk &node;
    };

    // ** schedule comptime ast **
    const schedules = comptime zensor.Scheduler.run(sum_node);
    std.debug.print("Schedules: {any}\n", .{schedules});

    // ** IR generator **
    var ir_generator = zensor.IRGenerator.init(allocator);
    defer ir_generator.deinit();
    const ir_block = try ir_generator.run(schedules[0]);
    std.debug.print("{}\n", .{ir_block});
}
