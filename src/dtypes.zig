const std = @import("std");

pub const DTypeNames = enum(u8) {
    Int32,
    Int64,
    Float32,
    Float64,

    pub fn isFloat(dtype: DTypeNames) bool {
        return switch (dtype) {
            .Float32, .Float64 => true,
            else => false,
        };
    }

    pub fn isInt(dtype: DTypeNames) bool {
        return !isFloat(dtype);
    }
};

pub const int32 = DataType{ .bits = 32, .name = .Int32 };
pub const int64 = DataType{ .bits = 64, .name = .Int64 };
pub const float32 = DataType{ .bits = 32, .name = .Float32 };
pub const float64 = DataType{ .bits = 64, .name = .Float64 };

pub fn FromNumpy(dtype: []const u8) ?DataType {
    if (std.mem.eql(u8, dtype, "<f4")) {
        return float32;
    }

    if (std.mem.eql(u8, dtype, "<f8")) {
        return float64;
    }

    if (std.mem.eql(u8, dtype, "<i4")) {
        return int32;
    }

    if (std.mem.eql(u8, dtype, "<i8")) {
        return int64;
    }

    return null;
}

pub const DataType = extern struct {
    const Self = @This();

    bits: u8,
    name: DTypeNames,

    pub fn ToBuiltin(comptime self: DataType) type {
        switch (self.name) {
            .Int32 => return i32,
            .Int64 => return i64,
            .Float32 => return f32,
            .Float64 => return f64,
        }
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;
        try writer.print("dtypes.{s}", .{@tagName(self.name)});
    }
};
