const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

const testing = std.testing;
const assert = std.debug.assert;

pub const Error = error{
    ShapeMismatch,
    CannotBroadcast,
    WrongType,
};

pub fn Tensor(comptime T: type) type {
    assert(@typeInfo(T) == .Float or @typeInfo(T) == .Int);

    return struct {
        const Self = @This();

        buffer: []u8,
        shape: []usize,
        strides: []usize,
        allocator: Allocator,

        pub usingnamespace Creation(T);
        pub usingnamespace Movement(T);
        pub usingnamespace Ops(T);

        fn init(allocator: Allocator, shape: []const usize) Allocator.Error!Self {
            return Self{
                .buffer = try determineBuffer(T, allocator, shape),
                .shape = try allocator.dupe(usize, shape),
                .strides = try determineStrides(allocator, shape),
                .allocator = allocator,
            };
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

        fn broadcastToShape(self: *Self, targetShape: []const usize) !void {
            if (self.shape.len == targetShape.len) {
                var same = true;
                for (self.shape, targetShape) |s, t| {
                    same = same and s == t;
                }
                if (same) {
                    return;
                }
            }

            const buffer = try determineBuffer(T, self.allocator, targetShape);
            const newData = mem.bytesAsSlice(T, buffer);

            const shape = try self.allocator.dupe(usize, targetShape);

            const strides = try determineStrides(self.allocator, targetShape);

            for (newData, 0..) |*elem, idx| {
                elem.* = self.data()[idx % self.data().len];
            }

            self.deinit();

            self.buffer = buffer;
            self.shape = shape;
            self.strides = strides;
        }
    };
}

pub fn Creation(comptime T: type) type {
    return struct {
        const Self = Tensor(T);

        pub fn empty(allocator: Allocator, dims: anytype) Allocator.Error!Self {
            const shape = getShape(dims);
            const self = try Self.init(allocator, shape);
            return self;
        }

        pub fn zeros(allocator: Allocator, dims: anytype) Allocator.Error!Self {
            return try Self.full(allocator, dims, 0);
        }

        pub fn ones(allocator: Allocator, dims: anytype) Allocator.Error!Self {
            return try Self.full(allocator, dims, 1);
        }

        pub fn full(allocator: Allocator, dims: anytype, value: T) Allocator.Error!Self {
            const self = try Self.empty(allocator, dims);
            @memset(self.data(), value);
            return self;
        }

        pub fn arange(allocator: Allocator, start: usize, stop: usize) Allocator.Error!Self {
            const shape = getShape(stop - start);
            const self = try Self.init(allocator, shape);

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

        pub fn fromOwnedSlice(allocator: Allocator, slice: anytype) Allocator.Error!Self {
            const shape = try shapeOfSlice(T, allocator, slice);
            errdefer allocator.free(shape);

            const buffer = try determineBuffer(T, allocator, shape);
            errdefer allocator.free(buffer);

            const strides = try determineStrides(allocator, shape);
            errdefer allocator.free(strides);

            const self = Self{
                .buffer = buffer,
                .shape = shape,
                .strides = strides,
                .allocator = allocator,
            };

            struct {
                pub fn copyData(_slice: anytype, _data: []align(1) T, _strides: []usize, offset: usize, depth: usize) void {
                    switch (@TypeOf(_slice)) {
                        []const T => {
                            @memcpy(_data[offset .. offset + _slice.len], _slice);
                        },
                        else => {
                            for (_slice, 0..) |_innerSlice, idx| {
                                copyData(_innerSlice, _data, _strides, offset + idx * _strides[depth], depth + 1);
                            }
                        },
                    }
                }
            }.copyData(slice, self.data(), self.strides, 0, 0);

            return self;
        }
    };
}

pub fn Movement(comptime T: type) type {
    return struct {
        const Self = Tensor(T);

        pub fn reshape(self: *Self, comptime dims: anytype) !void {
            const comptimeshape = getShape(dims);
            const shape = comptimeshape[0..];
            errdefer self.allocator.free(shape);

            var length: usize = 1;
            for (shape) |dim| {
                length *= dim;
            }

            if (length != self.data().len) {
                return Error.ShapeMismatch;
            }

            self.allocator.free(self.shape);
            self.allocator.free(self.strides);

            self.shape = try self.allocator.dupe(usize, shape);
            self.strides = try determineStrides(self.allocator, shape);
        }
    };
}

pub fn Ops(comptime T: type) type {
    return struct {
        const Self = Tensor(T);

        inline fn elementWiseOp(self: *Self, other: *Self, comptime op: fn (T, T) T) (Error || Allocator.Error)!Self {
            const shape = try broadcastShape(self.allocator, self.shape, other.shape);
            defer self.allocator.free(shape);
            errdefer self.allocator.free(shape);

            try self.broadcastToShape(shape);
            try other.broadcastToShape(shape);

            const result = try Self.init(self.allocator, shape);

            for (result.data(), self.data(), other.data()) |*r, s, o| {
                r.* = op(s, o);
            }

            return result;
        }

        pub fn add(self: *Self, other: *Self) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a + b;
                }
            }.op);
        }

        pub fn subtract(self: *Self, other: *Self) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a - b;
                }
            }.op);
        }

        pub fn multiply(self: *Self, other: *Self) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a * b;
                }
            }.op);
        }

        pub fn divide(self: *Self, other: *Self) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a / b;
                }
            }.op);
        }
    };
}

fn determineBuffer(comptime T: type, allocator: Allocator, shape: []const usize) ![]u8 {
    var size: usize = 1;

    for (shape) |dim| {
        size *= dim;
    }

    return try allocator.alloc(u8, size * @sizeOf(T) / @sizeOf(u8));
}

fn determineStrides(allocator: Allocator, shape: []const usize) ![]usize {
    var strides = try allocator.alloc(usize, shape.len);

    strides[shape.len - 1] = 1;
    var i = shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = shape[i] * strides[i];
    }

    return strides;
}

// follows numpy broadcasting rules
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

fn shapeFromStruct(comptime dims: anytype) [dims.len]usize {
    if (@typeInfo(@TypeOf(dims)) != .Struct) {
        @compileError("Dimensions must be a struct");
    }

    var shape: [dims.len]usize = undefined;
    for (dims, 0..) |dim, i| {
        shape[i] = switch (@TypeOf(dim)) {
            comptime_int => dim,
            usize => dim,
            else => @compileError("Dimensions must be of type comptime_int or usize"),
        };
    }
    return shape;
}

fn shapeFromInt(dim: anytype) [1]usize {
    if (@typeInfo(@TypeOf(dim)) != .Int and @typeInfo(@TypeOf(dim)) != .ComptimeInt) {
        @compileError("Dimensions must be an int");
    }

    return [_]usize{dim};
}

fn getShape(dims: anytype) []const usize {
    const array = switch (@typeInfo(@TypeOf(dims))) {
        .Struct => comptime shapeFromStruct(dims),
        .Int => shapeFromInt(dims),
        .ComptimeInt => shapeFromInt(dims),
        else => @compileError("Unsupported type. Expected struct or int"),
    };

    return array[0..];
}

fn shapeOfSlice(comptime T: type, allocator: Allocator, dims: anytype) ![]usize {
    const dim: usize = struct {
        fn countDims(slice: anytype, acc: usize) usize {
            if (@typeInfo(@TypeOf(slice)) == .Pointer) {
                if (slice.len == 0) {
                    return acc + 1;
                } else {
                    return countDims(slice[0], acc + 1);
                }
            } else if (@TypeOf(slice) == T) {
                return acc;
            }
        }
    }.countDims(dims, 0);

    const shape: []usize = try allocator.alloc(usize, dim);

    struct {
        fn setShape(_shape: []usize, _slice: anytype, idx: usize) void {
            if (@typeInfo(@TypeOf(_slice)) == .Pointer) {
                _shape[idx] = _slice.len;
                if (_slice.len > 0) {
                    setShape(_shape, _slice[0], idx + 1);
                }
            }
        }
    }.setShape(shape, dims, 0);

    return shape;
}

test {
    testing.refAllDecls(@This());
}
