const std = @import("std");

pub const DTypeKind = enum {
    Int,
    Float,

    pub fn isFloat(self: DTypeKind) bool {
        return self == .Float;
    }

    pub fn isInt(self: DTypeKind) bool {
        return self == .Int;
    }
};

pub const DType = union(enum) {
    Int32,
    Int64,
    Float32,
    Float64,
    BFloat16,

    pub fn kind(self: DType) DTypeKind {
        return switch (self) {
            .Int32, .Int64 => .Int,
            .Float32, .Float64, .BFloat16 => .Float,
        };
    }

    pub fn bits(self: DType) u8 {
        return switch (self) {
            .BFloat16 => 16,
            .Int32, .Float32 => 32,
            .Int64, .Float64 => 64,
        };
    }

    pub fn format(
        self: DType,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("dtypes.{s}", .{@tagName(self)});
    }

    pub fn ToBuiltin(comptime self: DType) type {
        return switch (self) {
            .Int32 => i32,
            .Int64 => i64,
            .Float32 => f32,
            .Float64 => f64,
            .BFloat16 => BFloat16,
        };
    }
};

pub const NumpyDType = struct {
    dtype: DType,
    endian: std.builtin.Endian,
};

pub fn FromNumpy(dtype: []const u8) ?NumpyDType {
    if (dtype.len < 2) return null;
    const endian_char = dtype[0];
    const type_str = dtype[1..];

    const endian = switch (endian_char) {
        '<' => .little,
        '>' => .big,
        '=' => @import("builtin").target.cpu.arch.endian(),
        else => return null,
    };

    const dtype_val: DType = if (std.mem.eql(u8, type_str, "f4"))
        .Float32
    else if (std.mem.eql(u8, type_str, "f8"))
        .Float64
    else if (std.mem.eql(u8, type_str, "i4"))
        .Int32
    else if (std.mem.eql(u8, type_str, "i8"))
        .Int64
    else if (std.mem.eql(u8, type_str, "b2"))
        .BFloat16
    else
        return null;

    return NumpyDType{ .dtype = dtype_val, .endian = endian };
}

pub const BFloat16 = packed struct {
    const Self = @This();

    value: u16,

    pub fn init(f: f32) Self {
        const bits = @as(u32, @bitCast(f));
        return .{ .value = @truncate(bits >> 16) };
    }

    pub fn toF32(self: Self) f32 {
        const bits = @as(u32, self.value) << 16;
        return @as(f32, @bitCast(bits));
    }

    pub fn add(self: Self, other: Self) Self {
        return Self.init(self.toF32() + other.toF32());
    }

    pub fn sub(self: Self, other: Self) Self {
        return Self.init(self.toF32() - other.toF32());
    }

    pub fn mul(self: Self, other: Self) Self {
        return Self.init(self.toF32() * other.toF32());
    }

    pub fn div(self: Self, other: Self) Self {
        return Self.init(self.toF32() / other.toF32());
    }
};
