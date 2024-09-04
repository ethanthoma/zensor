const std = @import("std");

const zensor = @import("zensor");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    try Mul(allocator);
}

fn Load(allocator: std.mem.Allocator) !void {
    const dtype = comptime zensor.dtypes.int64;

    const filename: []const u8 = "./examples/numpy.npy";
    var buffer: *zensor.RuntimeBuffer = blk: {
        var buffer = try zensor.RuntimeBuffer.Numpy.load(allocator, filename);
        break :blk &buffer;
    };
    defer buffer.deinit();

    const output = comptime blk: {
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

    var scheduler = zensor.Scheduler.init(allocator);
    try scheduler.register_buffer(output, buffer);
    try scheduler.mark_for_scheduling(output);

    const schedule = try scheduler.run(output);

    var ir_generator = zensor.IRGenerator.init(allocator);

    const ir_block = try ir_generator.run(schedule);
    std.debug.print("{}\n", .{ir_block});
}

fn Const(allocator: std.mem.Allocator) !void {
    const dtype = comptime zensor.dtypes.int64;

    const output = comptime blk: {
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

    var scheduler = zensor.Scheduler.init(allocator);
    try scheduler.mark_for_scheduling(output);

    const schedule = try scheduler.run(output);
    std.debug.print("{}\n", .{schedule});

    var ir_generator = zensor.IRGenerator.init(allocator);

    const ir_block = try ir_generator.run(schedule);
    std.debug.print("{}\n", .{ir_block});
}

fn Mul(allocator: std.mem.Allocator) !void {
    const dtype = comptime zensor.dtypes.int64;

    const filename: []const u8 = "./examples/numpy.npy";
    var buffer: *zensor.RuntimeBuffer = blk: {
        var buffer = try zensor.RuntimeBuffer.Numpy.load(allocator, filename);
        break :blk &buffer;
    };
    defer buffer.deinit();

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

    const output = comptime blk: {
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

    var scheduler = zensor.Scheduler.init(allocator);
    try scheduler.register_buffer(load_node, buffer);
    try scheduler.mark_for_scheduling(output);

    const schedule = try scheduler.run(output);
    std.debug.print("{}\n", .{schedule});

    var ir_generator = zensor.IRGenerator.init(allocator);

    const ir_block = try ir_generator.run(schedule);
    std.debug.print("{}\n", .{ir_block});
}
