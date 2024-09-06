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

pub fn FromNumpy(dtype: []const u8) ?DType {
    if (std.mem.eql(u8, dtype, "<f4")) {
        return .Float32;
    }

    if (std.mem.eql(u8, dtype, "<f8")) {
        return .Float64;
    }

    if (std.mem.eql(u8, dtype, "<i4")) {
        return .Int32;
    }

    if (std.mem.eql(u8, dtype, "<i8")) {
        return .Int64;
    }

    if (std.mem.eql(u8, dtype, "<b2")) {
        return .BFloat16;
    }

    return null;
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
