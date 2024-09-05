const std = @import("std");

const ast = @import("../ast.zig");
const dtypes = @import("../dtypes.zig");
const ir = @import("../ir.zig");
const Scheduler = @import("./Scheduler.zig");
const Schedule = @import("./Schedule.zig");
const Step = ir.Step;

const BufferID = usize;

const IRBufferContext = struct {
    step: Step,
    idx: BufferID,
};

const Range = struct {
    start: u32,
    end: u32,
};

const LoopContext = struct {
    range: Range,
    step: Step,
};

const Context = struct {
    allocator: std.mem.Allocator,
    node_map: std.AutoHashMap(*const ast.Node, u32),
    buffers: std.StringHashMap(IRBufferContext),
    loops: std.ArrayList(LoopContext),

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{
            .allocator = allocator,
            .node_map = std.AutoHashMap(*const ast.Node, u32).init(allocator),
            .buffers = std.StringHashMap(IRBufferContext).init(allocator),
            .loops = std.ArrayList(LoopContext).init(allocator),
        };
    }

    pub fn deinit(self: *Context) void {
        self.node_map.deinit();
        self.buffers.deinit();
        self.loops.deinit();
    }
};

const Self = @This();

pub fn run(allocator: std.mem.Allocator, schedule: *const Schedule) !ir.IRBlock {
    var context = Context.init(allocator);
    defer context.deinit();

    var block = ir.IRBlock.init(allocator);

    try define_global_buffers(&block, schedule, &context);

    for (schedule.nodes) |node| {
        try process_node(node, &block, &context);
    }

    while (context.loops.items.len > 0) {
        try close_loop(&block, &context);
    }

    return block;
}

fn define_global_buffers(block: *ir.IRBlock, schedule: *const Schedule, context: *Context) !void {
    for (schedule.global_buffers) |buffer_ctx| {
        if (!context.buffers.contains(buffer_ctx.name)) {
            const step = try block.append(
                .DEFINE_GLOBAL,
                .Pointer,
                null,
                .{
                    .idx = @intCast(buffer_ctx.idx),
                    .name = buffer_ctx.name,
                    .writable = buffer_ctx.writable,
                },
            );
            try context.buffers.put(buffer_ctx.name, .{ .idx = buffer_ctx.idx, .step = step });
        }
    }
}

fn process_node(node: *const ast.Node, block: *ir.IRBlock, context: *Context) !void {
    switch (node.op) {
        .Const => {
            const output = try block.append(
                .CONST,
                if (node.dtype.name.isInt()) .Int else .Float,
                null,
                node.arg.Const.value,
            );

            try context.node_map.put(node, output);
        },
        .Load => {
            const output = context.buffers.get(node.arg.Load.name).?.step;
            try context.node_map.put(node, output);
        },
        .Mul => {
            const output = try handle_binary_op(.Mul, node, block, context);
            try context.node_map.put(node, output);
        },
        .Store => {
            const loop_index = try render_loop(node, block, context);
            const buffer = context.buffers.get(node.arg.Store.name).?.step;
            const offset = try get_offset(node, loop_index);

            const input = context.node_map.get(node.input.Store[0]).?;
            const value = try get_value(
                node.input.Store[0],
                block,
                offset,
                input,
                context,
            );

            const output = try block.append(
                .STORE,
                null,
                try block.allocator.dupe(u32, &[_]u32{
                    buffer,
                    offset,
                    value,
                }),
                {},
            );

            try context.node_map.put(node, output);
        },
        .Sum => {
            const output = try handle_reduce_op(.Sum, node, block, context);
            try context.node_map.put(node, output);
        },
    }
}

fn render_above_scope(block: *ir.IRBlock, step: Step, context: *Context) !Step {
    const loop_ctx = context.loops.pop();
    const loop_index = loop_ctx.step;

    // remove step from original position
    const step_node = block.nodes.orderedRemove(step);

    // move step to above loop_index
    try block.nodes.insert(loop_index, step_node);

    // update IR node inputs
    for (block.nodes.items) |*ir_node| {
        if (ir_node.*.step == step) {
            ir_node.*.step = loop_index;
        } else if (ir_node.*.step >= loop_index) {
            ir_node.*.step += 1;
        }

        if (ir_node.*.inputs) |*inputs| {
            for (inputs.*) |*input| {
                if (input.* == step) {
                    input.* = loop_index;
                } else if (input.* >= loop_index) {
                    input.* += 1;
                }
            }
        }
    }

    // update node_map
    var iter = context.node_map.valueIterator();
    while (iter.next()) |step_ptr| {
        if (step_ptr.* == step) {
            step_ptr.* = loop_index;
        } else if (step_ptr.* >= loop_index) {
            step_ptr.* += 1;
        }
    }

    // update loops
    try context.loops.append(.{
        .range = loop_ctx.range,
        .step = loop_index + 1,
    });

    return loop_index;
}

fn get_range(node: *const ast.Node) Range {
    // input range
    switch (node.op) {
        inline else => |op| {
            const children = @field(node.input, @tagName(op));
            if (@typeInfo(@TypeOf(children)) == .Array) {
                return .{
                    .start = 0,
                    .end = children[0].view.size,
                };
            }
        },
    }

    // output range
    return .{
        .start = 0,
        .end = node.view.size,
    };
}

fn render_loop(node: *const ast.Node, block: *ir.IRBlock, context: *Context) !Step {
    const range = get_range(node);

    // check if in loop
    if (context.loops.getLastOrNull()) |loop_context| {
        // same iteration, return current loop index
        if (range.start == loop_context.range.start and range.end == loop_context.range.end) {
            return loop_context.step;
        } else {
            // different iteration, end loop
            try close_loop(block, context);
        }
    }

    // range of length one, return as const
    if (range.end - range.start == 1) {
        return try block.append(
            .CONST,
            .Int,
            null,
            try std.fmt.allocPrint(block.allocator, "{}", .{range.start}),
        );
    }

    const start_of_range = try block.append(
        .CONST,
        .Int,
        null,
        try std.fmt.allocPrint(block.allocator, "{}", .{range.start}),
    );

    const end_of_range = try block.append(
        .CONST,
        .Int,
        null,
        try std.fmt.allocPrint(block.allocator, "{}", .{range.end}),
    );

    const loop_index = try block.append(
        .LOOP,
        .Int,
        try block.allocator.dupe(u32, &[_]u32{
            start_of_range,
            end_of_range,
        }),
        {},
    );

    try context.loops.append(.{
        .range = range,
        .step = loop_index,
    });

    return loop_index;
}

fn close_loop(block: *ir.IRBlock, context: *Context) !void {
    if (context.loops.getLastOrNull()) |loop_context| {
        _ = try block.append(
            .ENDLOOP,
            null,
            try block.allocator.dupe(u32, &[_]u32{loop_context.step}),
            {},
        );

        _ = context.loops.pop();
    }
}

fn handle_binary_op(
    comptime op: ast.Operation,
    node: *const ast.Node,
    block: *ir.IRBlock,
    context: *Context,
) !Step {
    if (comptime op.AsOperationType() != .Binary) {
        @compileError("handle_binary_op only handles binary ops");
    }

    const loop_index = try render_loop(node, block, context);

    const offset = try get_offset(node, loop_index);

    const node_inputs = @field(node.input, @tagName(op));

    const lhs_step = context.node_map.get(node_inputs[0]).?;
    const lhs = try get_value(
        node_inputs[0],
        block,
        offset,
        lhs_step,
        context,
    );

    const rhs_step = context.node_map.get(node_inputs[1]).?;
    const rhs = try get_value(
        node_inputs[1],
        block,
        offset,
        rhs_step,
        context,
    );

    const alu_op = switch (op) {
        .Mul => .Mul,
        else => unreachable,
    };

    const output = try block.append(
        .ALU,
        if (node.dtype.name.isInt()) .Int else .Float,
        try block.allocator.dupe(u32, &[_]u32{ lhs, rhs }),
        alu_op,
    );

    return output;
}

fn handle_reduce_op(
    comptime op: ast.Operation,
    node: *const ast.Node,
    block: *ir.IRBlock,
    context: *Context,
) !Step {
    if (comptime op.AsOperationType() != .Reduce) {
        @compileError("handle_reduce_op only handles reduce ops");
    }

    const default_value = switch (op) {
        .Sum => 0,
        else => unreachable,
    };

    const acc = blk: {
        const acc = try block.append(
            .DEFINE_ACC,
            if (node.dtype.name.isInt()) .Int else .Float,
            null,
            try std.fmt.allocPrint(block.allocator, "{}", .{default_value}),
        );

        break :blk try render_above_scope(block, acc, context);
    };

    const loop_index = try render_loop(node, block, context);
    const offset = try get_offset(node, loop_index);

    const node_input = @field(node.input, @tagName(op))[0];
    const input = context.node_map.get(@field(node.input, @tagName(op))[0]).?;

    const reduce_input = try get_value(
        node_input,
        block,
        offset,
        input,
        context,
    );

    const alu_op = switch (op) {
        .Sum => .Add,
        else => unreachable,
    };

    const reduced = try block.append(
        .ALU,
        if (node.dtype.name.isInt()) .Int else .Float,
        try block.allocator.dupe(u32, &[_]u32{
            acc,
            reduce_input,
        }),
        alu_op,
    );

    const phi = try block.append(
        .PHI,
        if (node.dtype.name.isInt()) .Int else .Float,
        try block.allocator.dupe(u32, &[_]u32{
            acc,
            reduced,
            loop_index,
        }),
        {},
    );

    return phi;
}

// TODO: local buffer access
fn get_value(node: *const ast.Node, block: *ir.IRBlock, offset: Step, input: Step, context: *Context) !Step {
    switch (node.op) {
        .Load => {
            const buffer = context.buffers.get(node.arg.Load.name).?.step;

            const load = try block.append(
                .LOAD,
                if (node.dtype.name.isInt()) .Int else .Float,
                try block.allocator.dupe(u32, &[_]u32{ buffer, offset }),
                {},
            );

            return load;
        },
        else => {
            return input;
        },
    }
}

// TODO: should collapse view changes like permute, etc
fn get_offset(node: *const ast.Node, loop_index: Step) !Step {
    _ = node;
    return loop_index;
}
