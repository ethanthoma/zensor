const std = @import("std");
const FieldType = std.meta.FieldType;

const dtypes = @import("./dtypes.zig");
const view = @import("./view.zig");

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
    dtype: dtypes.DataType,

    const Self = @This();

    pub fn init(
        comptime op: Operation,
        arg: FieldType(Operation.Arg, op),
        input: FieldType(Operation.Input, op),
        anyview: *const view.AnyView,
        dtype: dtypes.DataType,
    ) Self {
        return Self{
            .op = op,
            .arg = @unionInit(Operation.Arg, @tagName(op), arg),
            .input = @unionInit(Operation.Input, @tagName(op), input),
            .view = anyview,
            .dtype = dtype,
        };
    }

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

        const allocator = if (!@inComptime()) blk: {
            var stack = std.heap.stackFallback(1024, std.heap.page_allocator);
            break :blk stack.get();
        } else null;

        try self.format_helper(allocator, writer, 0, false, "", offset);
    }

    fn format_helper(
        self: Self,
        allocator: ?std.mem.Allocator,
        writer: anytype,
        count: usize,
        is_last: bool,
        prefix: []const u8,
        offset: []const u8,
    ) !void {
        // print prefix
        try writer.writeAll(offset);
        try writer.print("{d} {s}", .{ count, prefix });
        if (count > 0) {
            try writer.writeAll(if (is_last) "┗━" else "┣━");
        }

        const new_prefix = if (count == 0) "" else if (is_last) "  " else "│ ";

        const p = try merge_two_str(allocator, prefix, new_prefix);
        defer p.free();

        const full_prefix = p.merged_strs;

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
                            allocator,
                            writer,
                            count + i,
                            i == input.len,
                            full_prefix,
                            offset,
                        );
                    }
                }

                break :blk;
            }
        }
    }

    const MergedStringResult = struct {
        allocator: ?std.mem.Allocator,
        merged_strs: []const u8,

        pub fn free(self: @This()) void {
            if (self.allocator) |allocator| {
                allocator.free(self.merged_strs);
            }
        }
    };

    fn merge_two_str(allocator: ?std.mem.Allocator, one: []const u8, two: []const u8) !MergedStringResult {
        if (allocator) |value| {
            const merged_strs = try std.fmt.allocPrint(value, "{s}{s}", .{ one, two });
            return .{ .allocator = value, .merged_strs = merged_strs };
        }
        if (@inComptime()) {
            const merged_strs = std.fmt.comptimePrint("{s}{s}", .{ one, two });
            return .{ .allocator = null, .merged_strs = merged_strs };
        }

        @panic("Must be comptime known strings or with a defined allocator");
    }
};
