const std = @import("std");

const ir = @import("./ir.zig");
const ast = @import("./ast.zig");
const Scheduler = @import("./Scheduler.zig");

const Self = @This();

allocator: std.mem.Allocator,
block: ir.IRBlock,
node_map: std.AutoHashMap(*const ast.Node, u32),
buffer_map: std.AutoHashMap(ast.BufferID, u32),
reduce_acc_map: std.AutoHashMap(*const ast.Node, u32),

pub fn init(allocator: std.mem.Allocator) Self {
    return .{
        .allocator = allocator,
        .block = undefined,
        .node_map = std.AutoHashMap(*const ast.Node, u32).init(allocator),
        .buffer_map = std.AutoHashMap(ast.BufferID, u32).init(allocator),
        .reduce_acc_map = std.AutoHashMap(*const ast.Node, u32).init(allocator),
    };
}

pub fn deinit(self: *Self) void {
    self.node_map.deinit();
    self.buffer_map.deinit();
    self.reduce_acc_map.deinit();
}

pub fn run(self: *Self, comptime schedule: Scheduler.Schedule) !ir.IRBlock {
    self.block = ir.IRBlock.init(self.allocator);

    try self.define_global_buffers(schedule);
    try self.define_reduce_accumulators(schedule);

    const start_of_range = try self.block.append(
        .CONST,
        .Int,
        null,
        try std.fmt.allocPrint(self.allocator, "{}", .{0}),
    );

    const end_of_range = try self.block.append(
        .CONST,
        .Int,
        null,
        try std.fmt.allocPrint(self.allocator, "{}", .{schedule.nodes[0].view.size}),
    );

    const loop_index = try self.block.append(
        .LOOP,
        .Int,
        try self.allocator.dupe(u32, &[_]u32{
            start_of_range,
            end_of_range,
        }),
        {},
    );

    inline for (schedule.nodes) |node| {
        try self.process_node(node, loop_index);
    }

    _ = try self.block.append(
        .ENDLOOP,
        null,
        try self.allocator.dupe(u32, &[_]u32{loop_index}),
        {},
    );

    return self.block;
}

fn define_global_buffers(self: *Self, comptime schedule: Scheduler.Schedule) !void {
    inline for (schedule.global_buffers) |buffer_ctx| {
        const buffer_id = buffer_ctx.id;
        if (!self.buffer_map.contains(buffer_id)) {
            const ir_node = try self.block.append(
                .DEFINE_GLOBAL,
                .Pointer,
                null,
                .{
                    .idx = buffer_id,
                    .name = try std.fmt.allocPrintZ(self.allocator, "data{}", .{buffer_id}),
                    .writable = buffer_ctx.writable,
                },
            );
            try self.buffer_map.put(buffer_id, ir_node);
        }
    }
}

fn isReduceOp(op: ast.Operations) bool {
    return switch (op) {
        .Sum => true,
        else => false,
    };
}

fn define_reduce_accumulators(self: *Self, comptime schedule: Scheduler.Schedule) !void {
    inline for (schedule.nodes) |node| {
        if (isReduceOp(node.op)) {
            const acc = try self.block.append(
                .DEFINE_ACC,
                if (node.dtype.name.isInt()) .Int else .Float,
                null,
                try std.fmt.allocPrint(self.allocator, "{}", .{0}),
            );

            try self.reduce_acc_map.put(node, acc);
        }
    }
}

fn process_node(self: *Self, comptime node: *const ast.Node, loop_index: u32) !void {
    var inputs = std.ArrayList(u32).init(self.allocator);
    defer inputs.deinit();

    switch (node.op) {
        .Mul, .Store, .Sum => {
            const node_inputs = @field(node.input, @tagName(node.op));
            for (node_inputs) |input| {
                const input_index = self.node_map.get(input).?;
                try inputs.append(input_index);
            }
        },
        .Load, .Const => {},
    }

    switch (node.op) {
        .Mul => {
            try self.binary_op(.Mul, node, inputs, loop_index);
        },
        .Load => {
            const output = self.buffer_map.get(node.arg.Load.buffer_id).?;
            try self.node_map.put(node, output);
        },
        .Store => {
            const buffer = self.buffer_map.get(node.arg.Store.buffer_id).?;

            const offset = try self.get_offset(node, loop_index);

            _ = try self.block.append(
                .STORE,
                null,
                try self.allocator.dupe(u32, &[_]u32{
                    buffer,
                    offset,
                    inputs.items[0],
                }),
                {},
            );
        },
        .Const => {
            const output = try self.block.append(
                .CONST,
                if (node.dtype.name.isInt()) .Int else .Float,
                null,
                node.arg.Const.value,
            );

            try self.node_map.put(node, output);
        },
        .Sum => {
            try self.reduce_op(.Sum, node, inputs, loop_index);
        },
    }
}

fn reduce_op(
    self: *Self,
    comptime op: enum { Sum },
    node: *const ast.Node,
    inputs: std.ArrayList(u32),
    loop_index: u32,
) !void {
    const acc = self.reduce_acc_map.get(node).?;

    const alu_op: std.meta.FieldType(ir.IROps.Arg, .ALU) = switch (op) {
        .Sum => .Add,
    };

    const input_value = switch (op) {
        .Sum => try self.get_value(node.input.Sum[0], loop_index, inputs.items[0]),
    };

    const updated_acc = try self.block.append(
        .ALU,
        if (node.dtype.name.isInt()) .Int else .Float,
        try self.allocator.dupe(u32, &[_]u32{ acc, input_value }),
        alu_op,
    );

    const phi_node = try self.block.append(
        .PHI,
        if (node.dtype.name.isInt()) .Int else .Float,
        try self.allocator.dupe(u32, &[_]u32{ acc, updated_acc, loop_index }),
        {},
    );

    try self.node_map.put(node, phi_node);
}

fn binary_op(
    self: *Self,
    comptime op: std.meta.FieldType(ir.IROps.Arg, .ALU),
    node: *const ast.Node,
    inputs: std.ArrayList(u32),
    loop_index: u32,
) !void {
    const lhs = try self.get_value(node.input.Mul[0], loop_index, inputs.items[0]);
    const rhs = try self.get_value(node.input.Mul[1], loop_index, inputs.items[1]);

    const output = try self.block.append(
        .ALU,
        if (node.dtype.name.isInt()) .Int else .Float,
        try self.allocator.dupe(u32, &[_]u32{
            lhs,
            rhs,
        }),
        op,
    );

    try self.node_map.put(node, output);
}

fn get_value(self: *Self, node: *const ast.Node, loop_index: u32, input_idx: u32) !u32 {
    if (node.op == .Load) {
        const offset = try self.get_offset(node, loop_index);

        return try self.block.append(
            .LOAD,
            if (node.dtype.name.isInt()) .Int else .Float,
            try self.allocator.dupe(u32, &[_]u32{ input_idx, offset }),
            {},
        );
    }

    return input_idx;
}

fn get_offset(self: *Self, node: *const ast.Node, loop_index: u32) !u32 {
    if (node.view.contiguous) {
        return loop_index;
    }

    var index = loop_index;

    var offset = try self.block.append(
        .CONST,
        .Int,
        null,
        try std.fmt.allocPrint(self.allocator, "{}", .{node.view.offset}),
    );

    for (0..node.view.rank) |i| {
        const stride = try self.block.append(
            .CONST,
            .Int,
            null,
            try std.fmt.allocPrint(self.allocator, "{}", .{node.view.strides[i]}),
        );

        const dim_size = try self.block.append(
            .CONST,
            .Int,
            null,
            try std.fmt.allocPrint(self.allocator, "{}", .{node.view.shape[i]}),
        );

        const dim_index = try self.block.append(
            .ALU,
            .Int,
            try self.allocator.dupe(u32, &[_]u32{ index, dim_size }),
            .Mod,
        );

        const dim_offset = try self.block.append(
            .ALU,
            .Int,
            try self.allocator.dupe(u32, &[_]u32{ dim_index, stride }),
            .Mul,
        );

        offset = try self.block.append(
            .ALU,
            .Int,
            try self.allocator.dupe(u32, &[_]u32{ offset, dim_offset }),
            .Add,
        );

        if (i < node.view.rank - 1) {
            index = try self.block.append(
                .ALU,
                .Int,
                try self.allocator.dupe(u32, &[_]u32{ index, dim_size }),
                .Div,
            );
        }
    }

    return offset;
}
