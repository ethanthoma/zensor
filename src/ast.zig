const std = @import("std");
const FieldType = std.meta.FieldType;

const dtypes = @import("./dtypes.zig");
const view = @import("./view.zig");

pub const Operations = enum {
    // Binary operations
    Mul,
    // Buffer operations
    Load,
    Store,
    // Initialization operations
    Const,
    // Reduce operations
    Sum,

    pub const Arg = union(Operations) {
        Mul: void,
        Load: struct {
            name: []const u8,
        },
        Store: struct {
            name: []const u8,
        },
        Const: struct {
            value: []const u8,
        },
        Sum: struct {
            dim: u32,
        },
    };

    pub const Input = union(Operations) {
        Mul: [2]*const Node,
        Load: void,
        Store: [1]*const Node,
        Const: void,
        Sum: [1]*const Node,
    };
};

pub const Node = struct {
    op: Operations,
    arg: Operations.Arg,
    input: Operations.Input,
    view: *const view.AnyView,
    dtype: dtypes.DataType,

    const Self = @This();

    pub fn init(
        comptime op: Operations,
        arg: FieldType(Operations.Arg, op),
        input: FieldType(Operations.Input, op),
        anyview: *const view.AnyView,
        dtype: dtypes.DataType,
    ) Self {
        return Self{
            .op = op,
            .arg = @unionInit(Operations.Arg, @tagName(op), arg),
            .input = @unionInit(Operations.Input, @tagName(op), input),
            .view = anyview,
            .dtype = dtype,
        };
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;

        try self.formatHelper(writer, 0, false, "");
    }

    fn formatHelper(
        self: Self,
        writer: anytype,
        count: u32,
        is_last: bool,
        prefix: []const u8,
    ) !void {
        try writer.print("{d} ", .{count});
        try writer.writeAll(prefix);
        if (count > 0) {
            try writer.writeAll(if (is_last) "┗━" else "┣━");
        }
        try writer.print("{s} ", .{@tagName(self.op)});

        switch (self.op) {
            .Mul => {},
            .Load => try writer.print("RuntimeBuffer(name={s}, dtype={}, shape={any})", .{
                self.arg.Load.name,
                self.dtype,
                self.view.shape[0..self.view.rank],
            }),
            .Store => try writer.print("RuntimeBuffer(name={s}, dtype={}, shape={any})", .{
                self.arg.Store.name,
                self.dtype,
                self.view.shape[0..self.view.rank],
            }),
            .Const => try writer.print("{s}", .{self.arg.Const.value}),
            .Sum => try writer.print("({d},)", .{self.arg.Sum.dim}),
        }

        const new_prefix = if (count == 0) "" else if (is_last) "  " else "│ ";
        const full_prefix = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}", .{ prefix, new_prefix });
        defer std.heap.page_allocator.free(full_prefix);

        switch (self.input) {
            .Mul => |children| {
                try writer.writeAll("\n");
                try children[0].formatHelper(writer, count + 1, false, full_prefix);

                try writer.writeAll("\n");
                try children[1].formatHelper(writer, count + 2, true, full_prefix);
            },
            .Store, .Sum => |child| {
                try writer.writeAll("\n");
                try child[0].formatHelper(writer, count + 1, true, full_prefix);
            },
            else => {},
        }
    }
};

pub const BufferID = u32;
