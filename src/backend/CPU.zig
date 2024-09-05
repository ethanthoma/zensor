const std = @import("std");

const ir = @import("../ir.zig");
const RuntimeBuffer = @import("../RuntimeBuffer.zig");
const dtypes = @import("../dtypes.zig");

/// this code is ugly and makes a lot of assumptions
/// a pretty ugly emulator running IR code directly
const CPU = @This();

const Error = std.fmt.ParseIntError || std.fmt.ParseFloatError || std.mem.Allocator.Error;

const Context = struct {
    allocator: std.mem.Allocator,
    block: ir.IRBlock,
    buffer_map: *std.AutoHashMap(usize, *RuntimeBuffer),
    values: std.AutoHashMap(ir.Step, ValueContext),

    pub fn init(
        allocator: std.mem.Allocator,
        block: ir.IRBlock,
        buffer_map: *std.AutoHashMap(usize, *RuntimeBuffer),
    ) @This() {
        return .{
            .allocator = allocator,
            .block = block,
            .buffer_map = buffer_map,
            .values = std.AutoHashMap(ir.Step, ValueContext).init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.values.deinit();
    }
};

const ValueContext = union(dtypes.DTypeNames) {
    Int32: i32,
    Int64: i64,
    Float32: f32,
    Float64: f64,

    // TODO: nuke this, should use actual type info
    // very important for when casting is added as an op
    fn asGeneric(self: ValueContext) f64 {
        return switch (self) {
            .Int32 => |v| @floatFromInt(v),
            .Int64 => |v| @floatFromInt(v),
            .Float32 => |v| v,
            .Float64 => |v| v,
        };
    }

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) @TypeOf(writer).Error!void {
        _ = fmt;
        _ = options;

        switch (self) {
            .Int32 => try writer.print("{d}", .{self.Int32}),
            .Int64 => try writer.print("{d}", .{self.Int64}),
            .Float32 => try writer.print("{}", .{self.Float32}),
            .Float64 => try writer.print("{}", .{self.Float64}),
        }
    }
};

pub fn run(allocator: std.mem.Allocator, block: ir.IRBlock, buffer_map: *std.AutoHashMap(usize, *RuntimeBuffer)) !void {
    var context = Context.init(allocator, block, buffer_map);
    defer context.deinit();

    var pc: u32 = 0;
    while (pc < block.len()) {
        pc += try execute_node(pc, &context);
    }
}

fn execute_node(
    pc: u32,
    context: *Context,
) Error!u32 {
    std.debug.print("PC: {d: >3}", .{pc});
    const node = context.block.nodes.items[pc];

    const pc_offset = switch (node.op) {
        .DEFINE_ACC => try define_acc(node, context),
        .CONST => try define_const(node, context),
        .LOOP => try loop(pc, node, context),
        .LOAD => try load(node, context),
        .ALU => try alu(node, context),
        .UPDATE => try update(node, context),
        .STORE => try store(node, context),
        else => blk: {
            std.debug.print("\n", .{});
            break :blk 1;
        },
    };
    return pc_offset;
}

fn define_acc(node: ir.IRNode, context: *Context) !u32 {
    const value_ctx: ValueContext = switch (node.dtype.?) {
        .Int => .{ .Int32 = try std.fmt.parseInt(i32, node.arg.DEFINE_ACC, 10) },
        .Float => .{ .Float32 = try std.fmt.parseFloat(f32, node.arg.DEFINE_ACC) },
        else => unreachable,
    };
    try context.values.put(node.step, value_ctx);

    std.debug.print("\tACC: {}\n", .{value_ctx});

    return 1;
}

fn define_const(node: ir.IRNode, context: *Context) !u32 {
    const value_ctx: ValueContext = switch (node.dtype.?) {
        .Int => .{ .Int32 = try std.fmt.parseInt(i32, node.arg.CONST, 10) },
        .Float => .{ .Float32 = try std.fmt.parseFloat(f32, node.arg.CONST) },
        else => unreachable,
    };
    try context.values.put(node.step, value_ctx);

    std.debug.print("\tCONST: {}\n", .{value_ctx});

    return 1;
}

fn get_instruction(pc: u32, context: *Context) ir.IRNode {
    return context.block.nodes.items[pc];
}

fn loop(pc: u32, node: ir.IRNode, context: *Context) Error!u32 {
    const start: u32 = @intFromFloat(context.values.get(node.inputs.?[0]).?.asGeneric());
    const end: u32 = @intFromFloat(context.values.get(node.inputs.?[1]).?.asGeneric());

    var loop_index = start;

    std.debug.print("\tLOOP: from {d} to {d}\n", .{ start, end });

    var offset: u32 = 0;
    while (loop_index < end) : (loop_index += 1) {
        try context.values.put(node.step, .{ .Int32 = @intCast(loop_index) });

        var inner_pc = pc + 1;
        while (!(get_instruction(inner_pc, context).op == .ENDLOOP and
            get_instruction(inner_pc, context).inputs.?[0] == node.step))
        {
            inner_pc += try execute_node(inner_pc, context);
            if (loop_index == start) {
                offset += 1;
            }
        }
    }

    return offset + 1;
}

fn load(node: ir.IRNode, context: *Context) !u32 {
    const buffer_idx = node.inputs.?[0];
    const buffer = context.buffer_map.get(buffer_idx).?;

    const index: u32 = @intFromFloat(context.values.get(node.inputs.?[1]).?.asGeneric());

    const value_ctx: ValueContext = switch (buffer.dtype.name) {
        .Int32 => .{ .Int32 = std.mem.bytesToValue(i32, buffer.get(index).?) },
        .Int64 => .{ .Int64 = std.mem.bytesToValue(i64, buffer.get(index).?) },
        .Float32 => .{ .Float32 = std.mem.bytesToValue(f32, buffer.get(index).?) },
        .Float64 => .{ .Float64 = std.mem.bytesToValue(f64, buffer.get(index).?) },
    };

    std.debug.print("\tLOAD: {} from buffer {} at {}\n", .{ value_ctx, buffer_idx, index });

    try context.values.put(node.step, value_ctx);

    return 1;
}

fn alu(node: ir.IRNode, context: *Context) !u32 {
    const lhs = context.values.get(node.inputs.?[0]).?.asGeneric();
    const rhs = context.values.get(node.inputs.?[1]).?.asGeneric();

    const result: ValueContext = switch (node.arg.ALU) {
        .Add => .{ .Float64 = lhs + rhs },
        .Div => .{ .Float64 = lhs / rhs },
        .Mod => .{ .Float64 = @mod(lhs, rhs) },
        .Mul => .{ .Float64 = lhs * rhs },
    };

    try context.values.put(node.step, result);

    std.debug.print("\tALU: {s}({}, {}) = {}\n", .{
        @tagName(node.arg.ALU),
        context.values.get(node.inputs.?[0]).?,
        context.values.get(node.inputs.?[1]).?,
        result,
    });

    return 1;
}

fn condition(node: ir.IRNode, context: *Context) bool {
    switch (node.op) {
        .LOOP => {
            const end: u32 = @intFromFloat(context.values.get(node.inputs.?[1]).?.asGeneric());
            const loop_index: u32 = @intFromFloat(context.values.get(node.step).?.asGeneric());

            return loop_index == end;
        },
        else => unreachable,
    }
}

// Maybe we want to use SSA for the IR?
fn phi(node: ir.IRNode, context: *Context) !u32 {
    std.debug.print("\tPHI: chose ", .{});

    const cond = condition(context.block.nodes.items[node.inputs.?[2]], context);

    const value = if (cond)
        context.values.get(node.inputs.?[0]).?
    else
        context.values.get(node.inputs.?[1]).?;

    std.debug.print("{s} branch ", .{if (cond) "left" else "right"});

    std.debug.print("with value {}\n", .{value});

    // store the value for this PHI node's step
    // pretty sure the whole point of SSA is that I only have to store once...?
    try context.values.put(node.step, value);

    return 1;
}

fn update(node: ir.IRNode, context: *Context) !u32 {
    const value = context.values.get(node.inputs.?[1]).?;

    const step_to_update = node.inputs.?[0];
    try context.values.put(step_to_update, value);

    try context.values.put(node.step, value);

    std.debug.print("\tUPDATE: value stored in step {} to {}\n", .{ step_to_update, value });

    return 1;
}

fn store(node: ir.IRNode, context: *Context) !u32 {
    const buffer_idx = node.inputs.?[0];
    const buffer = context.buffer_map.get(buffer_idx).?;

    const index: u32 = @intFromFloat(context.values.get(node.inputs.?[1]).?.asGeneric());

    const value = context.values.get(node.inputs.?[2]).?.asGeneric();

    std.debug.print("\tSTORE: {} into {} at {}\n", .{
        context.values.get(node.inputs.?[2]).?,
        buffer_idx,
        index,
    });

    switch (buffer.dtype.name) {
        .Int32 => {
            const item: i32 = @intFromFloat(value);
            buffer.set(index, &std.mem.toBytes(item));
        },
        .Int64 => {
            const item: i64 = @intFromFloat(value);
            buffer.set(index, &std.mem.toBytes(item));
        },
        .Float32 => {
            const item: f32 = @floatCast(value);
            buffer.set(index, &std.mem.toBytes(item));
        },
        .Float64 => {
            const item: f64 = value;
            buffer.set(index, &std.mem.toBytes(item));
        },
    }

    return 1;
}
