const std = @import("std");
const FieldType = std.meta.FieldType;

const dtypes = @import("../dtypes.zig");
const view = @import("../view.zig");

pub const OperationType = enum {
    Binary,
    Buffer,
    Initialization,
    Reduce,
};

pub const Operation = enum {
    // Binary operations
    Mul,
    // Buffer operations
    Load,
    Store,
    // Initialization operations
    Const,
    // Reduce operations
    Sum,

    pub fn AsOperationType(self: Operation) OperationType {
        return switch (self) {
            .Mul => .Binary,
            .Load, .Store => .Buffer,
            .Const => .Initialization,
            .Sum => .Reduce,
        };
    }

    pub fn has_children(self: Operation) bool {
        return switch (self) {
            .Load, .Const => false,
            else => true,
        };
    }

    pub const Arg = union(Operation) {
        Mul: void,
        Load: struct { name: []const u8 },
        Store: struct { name: []const u8 },
        Const: struct { value: []const u8 },
        Sum: struct { dim: u32 },
    };

    pub const Input = union(Operation) {
        Mul: [2]*const Node,
        Load: void,
        Store: [1]*const Node,
        Const: void,
        Sum: [1]*const Node,
    };
};

pub const Node = struct {
    op: Operation,
    arg: Operation.Arg,
    input: Operation.Input,
    view: *const view.AnyView,
    dtype: dtypes.DType,

    const Self = @This();

    pub fn init(
        comptime op: Operation,
        arg: FieldType(Operation.Arg, op),
        input: FieldType(Operation.Input, op),
        anyview: *const view.AnyView,
        dtype: dtypes.DType,
    ) Self {
        return Self{
            .op = op,
            .arg = @unionInit(Operation.Arg, @tagName(op), arg),
            .input = @unionInit(Operation.Input, @tagName(op), input),
            .view = anyview,
            .dtype = dtype,
        };
    }

    const LinkedStr = struct {
        content: []const u8,
        previous: ?*const LinkedStr,

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) @TypeOf(writer).Error!void {
            _ = options;

            try writer.writeAll(self.content);

            if (self.previous) |previous| {
                try writer.print("{" ++ fmt ++ "}", .{previous});
            }
        }
    };

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) @TypeOf(writer).Error!void {
        _ = options;

        const tab_count = comptime if (fmt.len != 0 and fmt[0] == 't')
            std.fmt.parseInt(u32, fmt[1..], 10) catch 0
        else
            0;

        const offset = "\t" ** tab_count;

        try self.format_helper(writer, 0, false, .{ .content = "", .previous = null }, offset);
    }

    fn format_helper(
        self: Self,
        writer: anytype,
        count: usize,
        is_last: bool,
        prefix: LinkedStr,
        offset: []const u8,
    ) !void {
        // print prefix
        try writer.writeAll(offset);
        try writer.print("{d} {s}", .{ count, prefix });
        if (count > 0) {
            try writer.writeAll(if (is_last) "┗━" else "┣━");
        }

        const new_prefix = LinkedStr{
            .content = if (count == 0) "" else if (is_last) "  " else "│ ",
            .previous = &prefix,
        };

        const fields = comptime std.meta.fields(Operation);
        inline for (fields) |field| blk: {
            if (@intFromEnum(self.op) == field.value) {
                const op = @field(Operation, field.name);
                const arg = @field(self.arg, field.name);
                const input = @field(self.input, field.name);

                // print node name
                try writer.print("{s} ", .{field.name});

                // print node info
                switch (comptime op.AsOperationType()) {
                    .Binary => {
                        try writer.print("", .{});
                    },
                    .Buffer => {
                        try writer.print(
                            "RuntimeBuffer(name={s}, dtype={}, shape={any})",
                            .{
                                arg.name,
                                self.dtype,
                                self.view.shape[0..self.view.rank],
                            },
                        );
                    },
                    .Initialization => {
                        try writer.print("{s}", .{arg.value});
                    },
                    .Reduce => {
                        try writer.print("({d},)", .{arg.dim});
                    },
                }

                // print children
                if (@typeInfo(@TypeOf(input)) == .Array) {
                    for (input, 1..) |child, i| {
                        try writer.writeAll("\n");
                        try child.format_helper(
                            writer,
                            count + i,
                            i == input.len,
                            new_prefix,
                            offset,
                        );
                    }
                }

                break :blk;
            }
        }
    }
};
