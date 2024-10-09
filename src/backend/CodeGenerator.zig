const std = @import("std");
const mem = std.mem;
const fmt = std.fmt;
const io = std.io;

const ir = @import("../compiler/ir.zig");

const Context = struct {
    block: *const ir.Block,
    cursor: usize = 0,
    writer: *std.ArrayList(u8).Writer,
    depth: usize = 0,
};

pub fn generate_kernel(allocator: mem.Allocator, kernel_name: []const u8, block: *const ir.Block) ![]const u8 {
    var code = std.ArrayList(u8).init(allocator);
    var writer = code.writer();

    var ctx = Context{
        .block = block,
        .cursor = 0,
        .writer = &writer,
        .depth = 0,
    };

    try writer.print("pub fn {s}(", .{kernel_name});

    try generate_kernel_parameters(&ctx);

    _ = try writer.write(") void {\n");

    ctx.depth += 1;

    while (ctx.cursor < block.nodes.items.len) {
        try generate_node(&ctx);
    }

    ctx.depth -= 1;

    _ = try writer.write("}");

    return code.toOwnedSlice();
}

fn generate_kernel_parameters(ctx: *Context) !void {
    var is_first = true;
    while (ctx.block.nodes.items[ctx.cursor].op == .DEFINE_GLOBAL) {
        if (!is_first) {
            try ctx.writer.writeAll(", ");
        }

        const node = ctx.block.nodes.items[ctx.cursor];
        const arg = node.arg.DEFINE_GLOBAL;

        try write_value(node, ctx);
        try ctx.writer.writeAll(": ");

        if (!arg.writable) {
            try ctx.writer.writeAll("const ");
        }

        try ctx.writer.writeAll("*RuntimeBuffer");

        try write_type(node, ctx);

        ctx.cursor += 1;
        is_first = false;
    }
}

fn write_type(node: ir.Node, ctx: *Context) !void {
    switch (node.dtype.?) {
        .Int => try ctx.writer.writeAll("u32"),
        .Float => try ctx.writer.writeAll("f32"),
        .Pointer => {},
    }
}

fn generate_node(ctx: *Context) !void {
    const node = ctx.block.nodes.items[ctx.cursor];

    switch (node.op) {
        .DEFINE_ACC => try generate_define_acc(node, ctx),
        .LOOP => try generate_loop(node, ctx),
        .LOAD => try generate_load(node, ctx),
        .ALU => try generate_alu(node, ctx),
        .UPDATE => try generate_update(node, ctx),
        .ENDLOOP => try generate_endloop(node, ctx),
        .STORE => try generate_store(node, ctx),
        else => {},
    }

    ctx.cursor += 1;
}

fn generate_define_acc(node: ir.Node, ctx: *Context) !void {
    try write_indent(ctx);
    try ctx.writer.writeAll("var ");
    try write_value(node, ctx);
    try ctx.writer.writeAll(": ");
    try write_type(node, ctx);
    try ctx.writer.print(" = {s};\n", .{node.arg.DEFINE_ACC});
}

fn write_indent(ctx: *Context) !void {
    for (0..ctx.depth) |_| {
        try ctx.writer.writeAll("\t");
    }
}

fn generate_loop(node: ir.Node, ctx: *Context) !void {
    const start = ctx.block.nodes.items[node.inputs.?[0]];
    const end = ctx.block.nodes.items[node.inputs.?[1]];

    try write_indent(ctx);
    try ctx.writer.writeAll("for (");
    try write_value(start, ctx);
    try ctx.writer.writeAll("..");
    try write_value(end, ctx);
    try ctx.writer.writeAll(") |");
    try write_value(node, ctx);
    try ctx.writer.writeAll("| {\n");
    ctx.depth += 1;
}

fn generate_load(node: ir.Node, ctx: *Context) !void {
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const index = ctx.block.nodes.items[node.inputs.?[1]];
    try write_indent(ctx);
    try ctx.writer.writeAll("const ");
    try write_value(node, ctx);
    try ctx.writer.writeAll(" = ");
    try write_value(buffer, ctx);
    try ctx.writer.writeAll(".get(");
    try write_value(index, ctx);
    try ctx.writer.writeAll(");\n");
}

fn write_value(node: ir.Node, ctx: *Context) !void {
    switch (node.op) {
        .DEFINE_GLOBAL => {
            try ctx.writer.print("data_{}", .{node.arg.DEFINE_GLOBAL.idx});
        },
        .CONST => try ctx.writer.print("{s}", .{node.arg.CONST}),
        .DEFINE_ACC => try ctx.writer.print("acc_{d}", .{node.step}),
        .LOAD => try ctx.writer.print("val_{d}", .{node.step}),
        .LOOP => try ctx.writer.print("idx_{d}", .{node.step}),
        .ALU => try ctx.writer.print("alu_{d}", .{node.step}),
        else => unreachable,
    }
}

fn generate_alu(node: ir.Node, ctx: *Context) !void {
    const lhs = ctx.block.nodes.items[node.inputs.?[0]];
    const rhs = ctx.block.nodes.items[node.inputs.?[1]];
    const op = switch (node.arg.ALU) {
        .Add => "+",
        .Mul => "*",
        .Div => "/",
        .Mod => "%",
    };
    try write_indent(ctx);
    try ctx.writer.writeAll("const ");
    try write_value(node, ctx);
    try ctx.writer.writeAll(" = ");
    try write_value(lhs, ctx);
    try ctx.writer.print(" {s} ", .{op});
    try write_value(rhs, ctx);
    try ctx.writer.writeAll(";\n");
}

fn generate_update(node: ir.Node, ctx: *Context) !void {
    const lhs = ctx.block.nodes.items[node.inputs.?[0]];
    const rhs = ctx.block.nodes.items[node.inputs.?[1]];
    try write_indent(ctx);
    try write_value(lhs, ctx);
    try ctx.writer.writeAll(" = ");
    try write_value(rhs, ctx);
    try ctx.writer.writeAll(";\n");
}

fn generate_endloop(_: ir.Node, ctx: *Context) !void {
    ctx.depth -= 1;
    try write_indent(ctx);
    try ctx.writer.writeAll("}\n");
}

fn generate_store(node: ir.Node, ctx: *Context) !void {
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const index = ctx.block.nodes.items[node.inputs.?[1]];
    const item = ctx.block.nodes.items[node.inputs.?[2]];
    try write_indent(ctx);
    try write_value(buffer, ctx);
    try ctx.writer.writeAll(".set(");
    try write_value(index, ctx);
    try ctx.writer.writeAll(", ");
    try write_value(item, ctx);
    try ctx.writer.writeAll(");\n");
}
