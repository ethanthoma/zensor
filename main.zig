const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

const testing = std.testing;
const assert = std.debug.assert;

pub const Error = error{
    ShapeMismatch,
    CannotBroadcast,
};

pub fn Tensor(comptime T: type) type {
    assert(@typeInfo(T) == .Float or @typeInfo(T) == .Int);

    return struct {
        const Self = @This();

        buffer: []u8,
        shape: []const usize,
        strides: []usize,
        allocator: Allocator,

        fn init(allocator: Allocator, shape: []const usize) Allocator.Error!Self {
            const buffer = try determineBuffer(allocator, shape);

            const _shape = try allocator.alloc(usize, shape.len);
            @memcpy(_shape, shape);

            const strides = try determineStrides(allocator, shape);

            return Self{
                .buffer = buffer,
                .shape = _shape,
                .strides = strides,
                .allocator = allocator,
            };
        }

        fn determineBuffer(allocator: Allocator, shape: []const usize) ![]u8 {
            var size: usize = shape[shape.len - 1];
            var i = shape.len - 1;
            while (i > 0) {
                i -= 1;
                size *= shape[i];
            }

            return try allocator.alloc(u8, size * @sizeOf(T) / @sizeOf(u8));
        }

        fn determineStrides(allocator: Allocator, shape: []const usize) ![]usize {
            var strides = try allocator.alloc(usize, shape.len);

            strides[shape.len - 1] = 1;
            var i = shape.len - 1;
            while (i > 0) {
                i -= 1;
                strides[i] = shape[i + 1] * strides[i + 1];
            }

            return strides;
        }

        fn determineShape(allocator: Allocator, slice: anytype) ![]usize {
            if (@typeInfo(@TypeOf(slice)) != .Pointer) {
                const shape = try allocator.alloc(usize, 1);
                shape[0] = 1;
                return shape;
            }

            const dim: usize = struct {
                fn countDims(_slice: anytype, acc: usize) usize {
                    if (@typeInfo(@TypeOf(_slice)) == .Pointer) {
                        if (_slice.len == 0) {
                            return acc + 1;
                        } else {
                            return countDims(_slice[0], acc + 1);
                        }
                    } else {
                        return acc;
                    }
                }
            }.countDims(slice, 0);

            const shape = try allocator.alloc(usize, dim);

            struct {
                fn setShape(_shape: []usize, _slice: anytype, idx: usize) void {
                    if (@typeInfo(@TypeOf(_slice)) == .Pointer) {
                        _shape[idx] = _slice.len;
                        if (_slice.len > 0) {
                            setShape(_shape, _slice[0], idx + 1);
                        }
                    }
                }
            }.setShape(shape, slice, 0);

            return shape;
        }

        // TODO: check that slice type is the same as T
        pub fn fromOwnedSlice(allocator: Allocator, slice: anytype) Allocator.Error!Self {
            const shape = try determineShape(allocator, slice);

            const buffer = try determineBuffer(allocator, shape);
            const strides = try determineStrides(allocator, shape);

            const self = Self{
                .buffer = buffer,
                .shape = shape,
                .strides = strides,
                .allocator = allocator,
            };

            struct {
                pub fn copyData(_slice: anytype, _data: []align(1) T, _strides: []usize, offset: usize, index: usize) void {
                    if (@typeInfo(@TypeOf(_slice)) != .Pointer) {
                        unreachable();
                    }

                    if (_strides.len - 1 == index) {
                        for (0.._slice.len) |idx| {
                            if (@TypeOf(_slice[idx]) == T) {
                                _data[offset + idx] = _slice[idx];
                            }
                        }
                    } else {
                        for (_slice, 0..) |_innerSlice, idx| {
                            copyData(_innerSlice, _data, _strides, offset + idx * _strides[index], index + 1);
                        }
                    }
                }
            }.copyData(slice, self.data(), self.strides, 0, 0);

            return self;
        }

        fn zeros(allocator: Allocator, shape: []const usize) Allocator.Error!Self {
            const self = try Self.init(allocator, shape);

            @memset(self.buffer, 0);

            return self;
        }

        fn ones(allocator: Allocator, shape: []const usize) Allocator.Error!Self {
            const self = try Self.init(allocator, shape);

            @memset(self.data(), 1);

            return self;
        }

        fn arange(allocator: Allocator, start: usize, stop: usize) Allocator.Error!Self {
            const self = try Self.init(allocator, ([_]usize{stop - start})[0..]);

            const convertToT = struct {
                pub fn func(from: usize) T {
                    return switch (@typeInfo(T)) {
                        .Float => @floatFromInt(from),
                        .Int => @intCast(from),
                        else => unreachable,
                    };
                }
            }.func;

            const dat = self.data();
            const offset = convertToT(start);

            for (0..(stop - start)) |i| {
                dat[i] = offset + convertToT(i);
            }

            return self;
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.buffer);
            self.allocator.free(self.strides);
            self.allocator.free(self.shape);
        }

        pub fn data(self: Self) []align(1) T {
            return mem.bytesAsSlice(T, self.buffer);
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            _ = fmt;

            const formatData = struct {
                fn print(wrt: anytype, strides: @TypeOf(self.strides), shape: @TypeOf(self.shape), d: []align(1) T, currentShapeIndex: usize, offset: usize) !void {
                    for (0..currentShapeIndex + 1) |_| {
                        try wrt.print("\t", .{});
                    }

                    try wrt.print("[", .{});

                    // last shape
                    if (currentShapeIndex == shape.len - 1) {
                        for (0..shape[currentShapeIndex]) |idx| {
                            if (idx > 0) {
                                try wrt.print(", ", .{});
                            }
                            try wrt.print("{}", .{d[idx + offset]});
                        }
                    } else {
                        try wrt.print("\n", .{});

                        for (0..shape[currentShapeIndex]) |idx| {
                            if (idx > 0) {
                                try wrt.print(",\n", .{});
                            }
                            try print(wrt, strides, shape, d, currentShapeIndex + 1, offset + idx * strides[currentShapeIndex]);
                        }

                        try wrt.print("\n", .{});

                        for (0..currentShapeIndex + 1) |_| {
                            try wrt.print("\t", .{});
                        }
                    }

                    try wrt.print("]", .{});
                }
            }.print;

            try writer.print("Tensor(\n", .{});

            try writer.print("\ttype: {},\n", .{T});

            try writer.print("\tshape: [", .{});
            for (self.shape, 0..) |dim, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{}", .{dim});
            }
            try writer.print("],\n", .{});

            try writer.print("\tlength: {},\n", .{self.data().len});

            try writer.print("\tdata:\n", .{});
            try formatData(writer, self.strides, self.shape, self.data(), 0, 0);
            try writer.print("\n", .{});

            try writer.print(")", .{});
        }

        pub fn fill(self: Self, value: T) void {
            @memset(self.data(), value);
        }

        pub fn add(self: Self, other: Self) (Error || Allocator.Error)!Self {
            if (self.shape.len != other.shape.len) {
                return error.ShapeMismatch;
            }
            for (self.shape, other.shape) |s, o| {
                if (s != o) {
                    return error.ShapeMismatch;
                }
            }

            const result = try Self.init(
                self.allocator,
                self.shape,
            );

            for (0..self.data().len) |idx| {
                result.data()[idx] = self.data()[idx] + other.data()[idx];
            }

            return result;
        }

        // follows numpy broadcasting rules
        // techincally, this can be more advanced
        fn broadcastShape(allocator: Allocator, shape1: []const usize, shape2: []const usize) ![]usize {
            const max_len = @max(shape1.len, shape2.len);

            var result: []usize = try allocator.alloc(usize, max_len);
            errdefer allocator.free(result);

            for (0..max_len) |i| {
                const dim1 = if (i >= shape1.len) 1 else shape1[shape1.len - i - 1];
                const dim2 = if (i >= shape2.len) 1 else shape2[shape2.len - i - 1];

                if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
                    return Error.CannotBroadcast;
                }

                result[max_len - i - 1] = if (dim1 > dim2) dim1 else dim2;
            }

            return result[0..max_len];
        }

        fn broadcastToShape(self: Self, targetShape: []const usize) !Self {
            if (self.len == targetShape.len) {
                var same = true;
                for (self.shape, targetShape) |s, t| {
                    same = same and s == t;
                }
                if (same) {
                    return self;
                }
            }

            const result = try Self.init(self.allocator, targetShape);

            for (0..result.data().len) |idx| {
                result.data()[idx] = self.data()[idx % self.data().len];
            }
        }
    };
}

test "determineShape" {
    const allocator = std.testing.allocator;

    {
        const slice: u64 = 42;
        const result = try Tensor(u64).determineShape(allocator, slice);
        defer allocator.free(result);
        const expected = ([_]usize{1})[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }
    {
        const slice: []const u64 = &.{};
        const result = try Tensor(u64).determineShape(allocator, slice);
        defer allocator.free(result);
        const expected = ([_]usize{0})[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }
    {
        const slice: []const u64 = &.{ 1, 2, 3, 4, 5 };
        const result = try Tensor(u64).determineShape(allocator, slice);
        defer allocator.free(result);
        const expected = ([_]usize{5})[0..];
        try testing.expectEqualSlices(usize, expected, result);
    }
    {
        const slice: []const []const u64 = &.{ &.{ 1, 2, 3 }, &.{ 4, 5, 6 } };
        const result = try Tensor(u64).determineShape(allocator, slice);
        defer allocator.free(result);
        const expected = ([_]usize{ 2, 3 })[0..];
        try testing.expectEqualSlices(usize, expected, result);
    }
    {
        const slice: []const []const []const u64 = &.{ &.{ &.{ 1, 2 }, &.{ 3, 4 } }, &.{ &.{ 5, 6 }, &.{ 7, 8 } } };
        const result = try Tensor(u64).determineShape(allocator, slice);
        defer allocator.free(result);
        const expected = ([_]usize{ 2, 2, 2 })[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }
}

test "fromOwnedSlice" {
    const allocator = testing.allocator;

    {
        const slice: []const u64 = &.{ 1, 2, 3, 4, 5 };
        const tensor = try Tensor(u64).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]usize{5})[0..];
        try std.testing.expectEqualSlices(usize, expectedShape, tensor.shape);

        const expectedData = &[_]u64{ 1, 2, 3, 4, 5 };
        const actualData: []const u64 = @alignCast(@ptrCast(tensor.data()));
        try std.testing.expectEqualSlices(u64, expectedData, actualData);
    }
}

test "broadcastShape" {
    {
        const shape1 = ([_]usize{ 5, 4 })[0..];
        const shape2 = ([_]usize{1})[0..];
        const result = try Tensor(f64).broadcastShape(testing.allocator, shape1, shape2);
        defer testing.allocator.free(result);
        const expected = ([_]usize{ 5, 4 })[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }

    {
        const shape1 = ([_]usize{ 5, 4 })[0..];
        const shape2 = ([_]usize{4})[0..];
        const result = try Tensor(f64).broadcastShape(testing.allocator, shape1, shape2);
        defer testing.allocator.free(result);
        const expected = ([_]usize{ 5, 4 })[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }

    {
        const shape1 = ([_]usize{ 15, 3, 5 })[0..];
        const shape2 = ([_]usize{ 15, 1, 5 })[0..];
        const result = try Tensor(f64).broadcastShape(testing.allocator, shape1, shape2);
        defer testing.allocator.free(result);
        const expected = ([_]usize{ 15, 3, 5 })[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }

    {
        const shape1 = ([_]usize{ 15, 3, 5 })[0..];
        const shape2 = ([_]usize{ 3, 5 })[0..];
        const result = try Tensor(f64).broadcastShape(testing.allocator, shape1, shape2);
        defer testing.allocator.free(result);
        const expected = ([_]usize{ 15, 3, 5 })[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }

    {
        const shape1 = ([_]usize{ 15, 3, 5 })[0..];
        const shape2 = ([_]usize{ 3, 1 })[0..];
        const result = try Tensor(f64).broadcastShape(testing.allocator, shape1, shape2);
        defer testing.allocator.free(result);
        const expected = ([_]usize{ 15, 3, 5 })[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }

    {
        const shape1 = ([_]usize{3})[0..];
        const shape2 = ([_]usize{4})[0..];
        const result = Tensor(f64).broadcastShape(testing.allocator, shape1, shape2);
        const expected = Error.CannotBroadcast;

        try testing.expectError(expected, result);
    }

    {
        const shape1 = ([_]usize{ 2, 1 })[0..];
        const shape2 = ([_]usize{ 8, 4, 3 })[0..];
        const result = Tensor(f64).broadcastShape(testing.allocator, shape1, shape2);
        const expected = Error.CannotBroadcast;

        try testing.expectError(expected, result);
    }
}

test "broadcastToShape" {
    {}
}

test {
    testing.refAllDecls(@This());
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var A = try Tensor(u64).init(allocator, ([_]usize{ 2, 3 })[0..]);
    defer A.deinit();
    A.fill(4);
    std.debug.print("{}\n", .{A});

    const slice: []const []const u64 = &.{ &.{ 3, 2, 1 }, &.{ 3, 2, 1 } };
    const B = try Tensor(u64).fromOwnedSlice(allocator, slice);
    defer B.deinit();
    std.debug.print("{}\n", .{B});

    const C = try A.add(B);
    defer C.deinit();
    std.debug.print("{}\n", .{C});

    const D = try Tensor(u32).arange(allocator, 5, 12);
    defer D.deinit();
    std.debug.print("{}\n", .{D});
}
