const std = @import("std");

const compiler = @import("../Compiler.zig");
const ir = compiler.ir;

const RuntimeBuffer = @import("../RuntimeBuffer.zig");
const dtypes = @import("../dtypes.zig");

const LogBuffer = @import("./LogBuffer.zig");

/// this code is ugly and makes a lot of assumptions
/// a pretty ugly emulator running  code directly
const CPU = @This();

const Error = std.fmt.ParseIntError || std.fmt.ParseFloatError || std.mem.Allocator.Error;

const Context = struct {
    allocator: std.mem.Allocator,
    block: ir.Block,
    buffer_map: *std.AutoHashMap(usize, *RuntimeBuffer),
    values: std.AutoHashMap(ir.Step, ValueContext),
    log_buffer: LogBuffer,

    pub fn init(
        allocator: std.mem.Allocator,
        block: ir.Block,
        buffer_map: *std.AutoHashMap(usize, *RuntimeBuffer),
    ) @This() {
        return .{
            .allocator = allocator,
            .block = block,
            .buffer_map = buffer_map,
            .values = std.AutoHashMap(ir.Step, ValueContext).init(allocator),
            .log_buffer = LogBuffer.init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.values.deinit();
    }
};

const ValueContext = union(ir.DataTypes) {
    Int: i64,
    Float: f64,
    Pointer: *i64,

    // TODO: nuke this, should use actual type info
    // very important for when casting is added as an op
    fn asGeneric(self: ValueContext) f64 {
        return switch (self) {
            .Int => |v| @floatFromInt(v),
            .Float => |v| v,
            else => unreachable,
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
            .Int => try writer.print("{d}", .{self.Int}),
            .Float => try writer.print("{}", .{self.Float}),
            .Pointer => try writer.print("@{d}", .{@intFromPtr(self.Pointer)}),
        }
    }
};

pub fn run(allocator: std.mem.Allocator, block: ir.Block, buffer_map: *std.AutoHashMap(usize, *RuntimeBuffer)) !LogBuffer {
    var context = Context.init(allocator, block, buffer_map);
    defer context.deinit();

    var pc: u32 = 0;
    while (pc < block.len()) {
        pc += try execute_node(pc, &context);
    }

    return context.log_buffer;
}

fn execute_node(
    pc: u32,
    context: *Context,
) Error!u32 {
    try context.log_buffer.log("PC: {d: >3}", .{pc});
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
            try context.log_buffer.log("\n", .{});
            break :blk 1;
        },
    };
    return pc_offset;
}

fn define_acc(node: ir.Node, context: *Context) !u32 {
    const value_ctx: ValueContext = switch (node.dtype.?) {
        .Int => .{ .Int = try std.fmt.parseInt(i32, node.arg.DEFINE_ACC, 10) },
        .Float => .{ .Float = try std.fmt.parseFloat(f32, node.arg.DEFINE_ACC) },
        else => unreachable,
    };
    try context.values.put(node.step, value_ctx);

    try context.log_buffer.log("\tACC: {}\n", .{value_ctx});

    return 1;
}

fn define_const(node: ir.Node, context: *Context) !u32 {
    const value_ctx: ValueContext = switch (node.dtype.?) {
        .Int => .{ .Int = try std.fmt.parseInt(i32, node.arg.CONST, 10) },
        .Float => .{ .Float = try std.fmt.parseFloat(f32, node.arg.CONST) },
        else => unreachable,
    };
    try context.values.put(node.step, value_ctx);

    try context.log_buffer.log("\tCONST: {}\n", .{value_ctx});

    return 1;
}

fn get_instruction(pc: u32, context: *Context) ir.Node {
    return context.block.nodes.items[pc];
}

fn loop(pc: u32, node: ir.Node, context: *Context) Error!u32 {
    const start: u32 = @intFromFloat(context.values.get(node.inputs.?[0]).?.asGeneric());
    const end: u32 = @intFromFloat(context.values.get(node.inputs.?[1]).?.asGeneric());

    var loop_index = start;

    try context.log_buffer.log("\tLOOP: from {d} to {d}\n", .{ start, end });

    var offset: u32 = 0;
    while (loop_index < end) : (loop_index += 1) {
        try context.values.put(node.step, .{ .Int = @intCast(loop_index) });

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

fn load(node: ir.Node, context: *Context) !u32 {
    const buffer_idx = node.inputs.?[0];
    const buffer = context.buffer_map.get(buffer_idx).?;

    const index: u32 = @intFromFloat(context.values.get(node.inputs.?[1]).?.asGeneric());

    const value_ctx: ValueContext = switch (buffer.dtype) {
        .Int32 => .{ .Int = std.mem.bytesToValue(i32, buffer.get(index).?) },
        .Int64 => .{ .Int = std.mem.bytesToValue(i64, buffer.get(index).?) },
        .Float32 => .{ .Float = std.mem.bytesToValue(f32, buffer.get(index).?) },
        .Float64 => .{ .Float = std.mem.bytesToValue(f64, buffer.get(index).?) },
        else => unreachable,
    };

    try context.log_buffer.log("\tLOAD: {} from buffer {} at {}\n", .{ value_ctx, buffer_idx, index });

    try context.values.put(node.step, value_ctx);

    return 1;
}

fn alu(node: ir.Node, context: *Context) !u32 {
    const lhs = context.values.get(node.inputs.?[0]).?.asGeneric();
    const rhs = context.values.get(node.inputs.?[1]).?.asGeneric();

    const result: ValueContext = switch (node.arg.ALU) {
        .Add => .{ .Float = lhs + rhs },
        .Div => .{ .Float = lhs / rhs },
        .Mod => .{ .Float = @mod(lhs, rhs) },
        .Mul => .{ .Float = lhs * rhs },
    };

    try context.values.put(node.step, result);

    try context.log_buffer.log("\tALU: {s}({}, {}) = {}\n", .{
        @tagName(node.arg.ALU),
        context.values.get(node.inputs.?[0]).?,
        context.values.get(node.inputs.?[1]).?,
        result,
    });

    return 1;
}

fn condition(node: ir.Node, context: *Context) bool {
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
fn phi(node: ir.Node, context: *Context) !u32 {
    try context.log_buffer.log("\tPHI: chose ", .{});

    const cond = condition(context.block.nodes.items[node.inputs.?[2]], context);

    const value = if (cond)
        context.values.get(node.inputs.?[0]).?
    else
        context.values.get(node.inputs.?[1]).?;

    try context.log_buffer.log("{s} branch ", .{if (cond) "left" else "right"});

    try context.log_buffer.log("with value {}\n", .{value});

    // store the value for this PHI node's step
    // pretty sure the whole point of SSA is that I only have to store once...?
    try context.values.put(node.step, value);

    return 1;
}

fn update(node: ir.Node, context: *Context) !u32 {
    const value = context.values.get(node.inputs.?[1]).?;

    const step_to_update = node.inputs.?[0];
    try context.values.put(step_to_update, value);

    try context.log_buffer.log("\tUPDATE: value stored in step {} to {}\n", .{ step_to_update, value });

    return 1;
}

fn store(node: ir.Node, context: *Context) !u32 {
    const buffer_idx = node.inputs.?[0];
    const buffer = context.buffer_map.get(buffer_idx).?;

    const index: u32 = @intFromFloat(context.values.get(node.inputs.?[1]).?.asGeneric());

    const value = context.values.get(node.inputs.?[2]).?.asGeneric();

    try context.log_buffer.log("\tSTORE: {} into {} at {}\n", .{
        context.values.get(node.inputs.?[2]).?,
        buffer_idx,
        index,
    });

    switch (buffer.dtype) {
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
        else => unreachable,
    }

    return 1;
}
