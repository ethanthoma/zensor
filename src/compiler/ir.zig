const std = @import("std");

const RuntimeBuffer = @import("../RuntimeBuffer.zig");

pub const Step = u32;

pub const DataTypes = enum {
    Int,
    Float,
    Pointer,
};

pub const Ops = enum {
    DEFINE_GLOBAL,
    DEFINE_ACC,
    CONST,
    LOOP,
    LOAD,
    ALU,
    UPDATE,
    ENDLOOP,
    STORE,

    pub const Arg = union(Ops) {
        const Self = @This();

        DEFINE_GLOBAL: struct {
            idx: u32,
            writable: bool,

            pub fn format(
                self: @This(),
                comptime fmt: []const u8,
                options: std.fmt.FormatOptions,
                writer: anytype,
            ) !void {
                _ = options;
                _ = fmt;
                try writer.print("(", .{});
                try writer.print("{d}, ", .{self.idx});
                try writer.print("{any}", .{self.writable});
                try writer.print(")", .{});
            }
        },
        DEFINE_ACC: []const u8,
        CONST: []const u8,
        LOOP: void,
        LOAD: void,
        ALU: enum {
            Add,
            Div,
            Mod,
            Mul,

            pub fn format(
                self: @This(),
                comptime fmt: []const u8,
                options: std.fmt.FormatOptions,
                writer: anytype,
            ) !void {
                _ = options;
                _ = fmt;
                try writer.print("ALU.{s}", .{@tagName(self)});
            }
        },
        UPDATE: void,
        ENDLOOP: void,
        STORE: void,

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            _ = fmt;

            inline for (std.meta.fields(Self)) |field| {
                if (self == @field(Self, field.name)) {
                    if (@typeInfo(@TypeOf(@field(self, field.name))) == .void) {
                        try writer.print("None", .{});
                    } else if (@TypeOf(@field(self, field.name)) == []const u8) {
                        try writer.print("{s}", .{@field(self, field.name)});
                    } else {
                        try writer.print("{any}", .{@field(self, field.name)});
                    }
                    return;
                }
            }
        }
    };
};

pub const Node = struct {
    const Self = @This();

    step: Step,
    op: Ops,
    dtype: ?DataTypes,
    // this is convienient for quick iterations but probably should be changed to
    // work more like ast.Node does
    inputs: ?[]Step,
    arg: Ops.Arg,

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;
        try writer.print("{d: >4} ", .{self.step});

        try writer.print("{s: <16} ", .{@tagName(self.op)});

        if (self.dtype) |dtype| {
            try writer.print("{s: <16} ", .{@tagName(dtype)});
        } else {
            try writer.print("{s: <16} ", .{""});
        }

        var width: u64 = 2;
        try writer.print("[", .{});
        if (self.inputs) |inputs| {
            for (inputs, 0..) |input, i| {
                try writer.print("{d}", .{input});
                width += std.fmt.count("{d}", .{input});

                if (i != inputs.len - 1) {
                    try writer.print(", ", .{});
                    width += 2;
                }
            }
        }
        try writer.print("] ", .{});
        for (0..@max(16 - @min(width, 16), 0)) |_| {
            try writer.print(" ", .{});
        }

        if (@typeInfo(@TypeOf(self.arg)) == .void) {
            try writer.print("None", .{});
        } else {
            try writer.print("{any}", .{self.arg});
        }
    }
};

pub const Block = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node),

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .nodes = std.ArrayList(Node).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.nodes.deinit();
    }

    pub fn len(self: *const Self) usize {
        return self.nodes.items.len;
    }

    pub fn append(self: *Self, comptime op: Ops, dtype: ?DataTypes, inputs: ?[]Step, arg: std.meta.FieldType(Ops.Arg, op)) !Step {
        const step: Step = @intCast(self.len());

        try self.nodes.append(Node{
            .step = step,
            .op = op,
            .dtype = dtype,
            .inputs = inputs,
            .arg = @unionInit(Ops.Arg, @tagName(op), arg),
        });

        return step;
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;

        try writer.print("step ", .{});
        try writer.print("op name          ", .{});
        try writer.print("type             ", .{});
        try writer.print("input            ", .{});
        try writer.print("arg", .{});
        for (self.nodes.items) |node| {
            try writer.print("\n{}", .{node});
        }
    }
};
