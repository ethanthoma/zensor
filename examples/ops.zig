const std = @import("std");

const zensor = @import("zensor");
const Tensor = zensor.tensor;
const Node = zensor.Node;
const Operations = zensor.Operations;
const Shape = zensor.Shape;
const View = zensor.View;
const Numpy = zensor.Numpy;
const RuntimeBuffer = zensor.RuntimeBuffer;
const MemoryManager = zensor.MemoryManager;
const Scheduler = zensor.Scheduler;
const dtypes = zensor.dtypes;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // ** Load in Numpy tensor as u8 buffer **
    const n: RuntimeBuffer = try Numpy.load(allocator, "./examples/numpy.npy");

    // ** create comptime buffer id **
    const buffer_id = comptime MemoryManager.create_id(opaque {});

    // ** create memory manager that links comptime nodes with runtime buffers **
    var memory_manager: MemoryManager = MemoryManager.init();
    try memory_manager.set_buffer(allocator, buffer_id, n);

    // ** create comptime nodes **
    const load_node: *const Node = comptime blk: {
        const shape = &[_]u32{3};
        const view: *const zensor.AnyView = @ptrCast(&View(shape).init());
        const node = Node.init(
            Operations.Load,
            .{ .buffer_id = buffer_id },
            {},
            view,
            dtypes.int64,
        );

        break :blk &node;
    };

    const const_node: *const Node = comptime blk: {
        const shape = &[_]u32{3};
        const view: *const zensor.AnyView = @ptrCast(&View(shape).init());
        const node = Node.init(
            Operations.Const,
            .{ .value = std.fmt.comptimePrint("{}", .{4}) },
            {},
            view,
            dtypes.int64,
        );
        break :blk &node;
    };

    const mul_node: *const Node = comptime blk: {
        const shape = &[_]u32{3};
        const view: *const zensor.AnyView = @ptrCast(&View(shape).init());
        const node = Node.init(
            Operations.Mul,
            {},
            [2]*const Node{ load_node, const_node },
            view,
            dtypes.int64,
        );
        break :blk &node;
    };

    // ** schedule comptime node **
    const scheduler = Scheduler.init(allocator, memory_manager);
    var execution_plan = try scheduler.run(mul_node);
    defer execution_plan.deinit();

    std.debug.print("Execution Plan: {}\n", .{execution_plan});
}
