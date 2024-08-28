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
            buffer_id: BufferID,
        },
        Store: struct {
            buffer_id: BufferID,
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
        comptime anyview: *const view.AnyView,
        comptime dtype: dtypes.DataType,
    ) Self {
        return Self{
            .op = op,
            .arg = @unionInit(Operations.Arg, @tagName(op), arg),
            .input = @unionInit(Operations.Input, @tagName(op), input),
            .view = anyview,
            .dtype = dtype,
        };
    }
};

pub const BufferID = u32;
