const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;

pub const Error = error{
    ShapeMismatch,
    CannotBroadcast,
    WrongType,
};

pub usingnamespace @import("graph.zig");

pub var prng = std.Random.DefaultPrng.init(20240807);

pub fn manualSeed(seed: u64) void {
    prng = std.Random.DefaultPrng.init(seed);
}

pub fn Tensor(comptime dtype: dtypes.DType) type {
    const T = dtype.kind;
    return struct {
        const Self = @This();

        buffer: []u8,
        shape: []u64,
        strides: []u64,
        allocator: Allocator,

        pub usingnamespace Creation(dtype);
        pub usingnamespace Movement(dtype);
        pub usingnamespace Ops(dtype);

        pub fn init(allocator: Allocator, shape: []const u64) Allocator.Error!Self {
            return Self{
                .buffer = try determineBuffer(T, allocator, shape),
                .shape = try allocator.dupe(u64, shape),
                .strides = try determineStrides(allocator, shape),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.buffer);
            self.allocator.free(self.strides);
            self.allocator.free(self.shape);
        }

        pub fn clone(self: *Self) (Error || Allocator.Error)!Self {
            const buffer = try self.allocator.dupe(u8, self.buffer);
            errdefer self.allocator.free(buffer);

            const shape = try self.allocator.dupe(u64, self.shape);
            errdefer self.allocator.free(shape);

            const strides = try self.allocator.dupe(u64, self.strides);

            return Self{
                .buffer = buffer,
                .shape = shape,
                .strides = strides,
                .allocator = self.allocator,
            };
        }

        pub fn data(self: Self) []T {
            return @alignCast(mem.bytesAsSlice(T, self.buffer));
        }

        pub fn at(self: Self, index: u64) T {
            return self.data()[index];
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
                fn print(wrt: anytype, strides: @TypeOf(self.strides), shape: @TypeOf(self.shape), d: []T, currentShapeIndex: u64, offset: u64) !void {
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

            try writer.print("\ttype: {},\n", .{dtype});

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

        fn broadcastToShape(self: *Self, targetShape: []const u64) Allocator.Error!void {
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
            errdefer self.allocator.free(buffer);

            const newData = mem.bytesAsSlice(T, buffer);

            const shape = try self.allocator.dupe(u64, targetShape);
            errdefer self.allocator.free(shape);

            const strides = try determineStrides(self.allocator, targetShape);
            errdefer self.allocator.free(strides);

            for (newData, 0..) |*elem, idx| {
                elem.* = self.data()[idx % self.data().len];
            }

            self.deinit();

            self.buffer = buffer;
            self.shape = shape;
            self.strides = strides;
        }

        fn broadcast(self: *Self, other: anytype) (Error || Allocator.Error)!Self {
            var other_as_tensor: Self = if (@TypeOf(other) == *Self) try other.clone() else switch (@typeInfo(@TypeOf(other))) {
                .Float, .Int, .ComptimeInt => try Self.fullLike(self.allocator, self, other),
                .Pointer => try Self.fromOwnedSlice(self.allocator, other),
                else => return Error.WrongType,
            };
            errdefer other_as_tensor.deinit();

            const out_shape = try broadcastShape(self.allocator, self.shape, other_as_tensor.shape);
            defer self.allocator.free(out_shape);

            try self.broadcastToShape(out_shape);
            try other_as_tensor.broadcastToShape(out_shape);

            return other_as_tensor;
        }
    };
}

pub fn Creation(comptime dtype: dtypes.DType) type {
    return struct {
        const Self = Tensor(dtype);

        pub fn empty(allocator: Allocator, dims: anytype) (Error || Allocator.Error)!Self {
            const shape = getShape(dims);
            const self = try Self.init(allocator, shape);
            return self;
        }

        pub fn zeros(allocator: Allocator, dims: anytype) (Error || Allocator.Error)!Self {
            return try Self.full(allocator, dims, 0);
        }

        pub fn ones(allocator: Allocator, dims: anytype) (Error || Allocator.Error)!Self {
            return try Self.full(allocator, dims, 1);
        }

        pub fn full(allocator: Allocator, dims: anytype, value: anytype) (Error || Allocator.Error)!Self {
            const self = try Self.empty(allocator, dims);

            const val = dtype.castValueToKind(value);
            const bitCount = dtype.countBits(val);

            if (bitCount > dtype.bits) {
                std.debug.print("Value {} requires {} bytes, dtype is only {} bytes.\n", .{ val, (bitCount + 7) / 8, (dtype.bits + 7) / 8 });
                return Error.WrongType;
            }

            const bytes = try allocator.alloc(u8, (dtype.bits + 7) / 8);

            defer allocator.free(bytes);
            @memset(bytes, 0);

            dtype.bytesFromValue(&bytes, bitCount, val);

            for (0..self.buffer.len) |i| {
                self.buffer[i] = bytes[@mod(i, bytes.len)];
            }

            return self;
        }

        pub fn arange(allocator: Allocator, start: u64, stop: u64) Allocator.Error!Self {
            const shape = getShape(stop - start);
            const self = try Self.init(allocator, shape);

            const convertToT = struct {
                pub fn func(from: u64) dtype.kind {
                    return switch (@typeInfo(dtype.kind)) {
                        .Float => @floatFromInt(from),
                        .Int => @intCast(from),
                        // TODO: don't assume the struct is always BFloat16
                        .Struct => dtypes.BFloat16.fromF32(from),
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
            const shape = try shapeOfSlice(dtype.kind, allocator, slice);
            errdefer allocator.free(shape);

            const buffer = try determineBuffer(dtype.kind, allocator, shape);
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
                pub fn copyData(_slice: anytype, _data: []dtype.kind, _strides: []u64, offset: u64, depth: u64) void {
                    switch (@TypeOf(_slice)) {
                        []const dtype.kind => {
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

        pub fn fromScaler(self: *Self, value: anytype) (Error || Allocator.Error)!Self {
            const other = try Self.fullLike(self.allocator, self, value);
            other.data()[0] = value;
            return other;
        }

        pub fn fullLike(allocator: Allocator, tensor: *Self, value: dtype.kind) Allocator.Error!Self {
            const self = try Self.init(allocator, tensor.shape);
            @memset(self.data(), value);
            return self;
        }

        pub fn rand(allocator: Allocator, dims: anytype) Allocator.Error!Self {
            const shape = getShape(dims);
            const self = try Self.init(allocator, shape);
            for (self.data()) |*elem| {
                elem.* = switch (@typeInfo(dtype.kind)) {
                    .Float => blk: {
                        if (@typeInfo(dtype.kind).Float.bits < 32) {
                            const value = prng.random().float(f32);
                            break :blk @as(dtype.kind, @floatCast(@trunc(value)));
                        }
                        // Only support f32 and f64
                        break :blk prng.random().float(dtype.kind);
                    },
                    .Int => prng.random().int(dtype.kind),
                    else => unreachable,
                };
            }
            return self;
        }

        pub fn randInt(allocator: Allocator, dims: anytype, low: dtype.kind, high: dtype.kind) Allocator.Error!Self {
            if (@typeInfo(dtype.kind) != .Int) {
                return Error.WrongType;
            }

            const shape = getShape(dims);
            const self = try Self.init(allocator, shape);
            for (self.data()) |*elem| {
                elem.* = prng.random().intRangeLessThan(dtype.kind, low, high);
            }
            return self;
        }
    };
}

pub fn Movement(comptime dtype: dtypes.DType) type {
    return struct {
        const Self = Tensor(dtype);

        // TODO: add support for -1
        pub fn reshape(self: *Self, comptime dims: anytype) (Error || Allocator.Error)!void {
            const shape = getShape(dims);
            errdefer self.allocator.free(shape);

            var length: u64 = 1;
            for (shape) |dim| {
                length *= dim;
            }

            if (length != self.data().len) {
                return Error.ShapeMismatch;
            }

            self.allocator.free(self.shape);
            self.allocator.free(self.strides);

            self.shape = try self.allocator.dupe(u64, shape);
            // user should defer free self so no need to free shape if strides fail
            self.strides = try determineStrides(self.allocator, shape);
        }
    };
}

pub fn Ops(comptime dtype: dtypes.DType) type {
    const T = dtype.kind;
    return struct {
        const Self = Tensor(dtype);

        inline fn elementWiseOp(self: *Self, other: anytype, comptime op: fn (T, T) T) (Error || Allocator.Error)!Self {
            const broadcasted_other = try self.broadcast(other);
            defer broadcasted_other.deinit();

            const result = try Self.init(self.allocator, broadcasted_other.shape);

            for (result.data(), self.data(), broadcasted_other.data()) |*r, s, o| {
                r.* = op(s, o);
            }

            return result;
        }

        pub fn add(self: *Self, other: anytype) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a + b;
                }
            }.op);
        }

        pub fn sub(self: *Self, other: anytype) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a - b;
                }
            }.op);
        }

        pub fn mul(self: *Self, other: anytype) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a * b;
                }
            }.op);
        }

        pub fn div(self: *Self, other: anytype) (Error || Allocator.Error)!Self {
            return self.elementWiseOp(other, struct {
                fn op(a: T, b: T) T {
                    return a / b;
                }
            }.op);
        }

        pub fn matmul(self: *Self, other: *Self) (Error || Allocator.Error)!Self {
            if (self.shape.len != 2 or other.shape.len != 2) {
                return Error.ShapeMismatch;
            }
            if (self.shape[1] != other.shape[0]) {
                return Error.ShapeMismatch;
            }

            const m = self.shape[0];
            const n = other.shape[0];

            var result = try Self.zeros(self.allocator, .{ m, n });
            errdefer result.deinit();

            result.matmuladd(self, other);

            return result;
        }

        // Define block sizes
        // These are arbitrary values that seem to work well
        const dtypeSize = @sizeOf(dtype.kind);
        const M_BLOCK = 2048 / dtypeSize;
        const N_BLOCK = 2048 / dtypeSize;
        const K_BLOCK = 2048 / dtypeSize;

        threadlocal var a_local: [M_BLOCK * K_BLOCK]T align(@alignOf(T)) = undefined;
        threadlocal var b_local: [K_BLOCK * N_BLOCK]T align(@alignOf(T)) = undefined;
        threadlocal var c_local: [M_BLOCK * N_BLOCK]T align(@alignOf(T)) = undefined;

        pub fn matmuladd(c: *Self, a: *Self, b: *Self) (Error || Allocator.Error)!*Self {
            @setFloatMode(.optimized);
            if (a.shape.len != 2 or b.shape.len != 2 or c.shape.len != 2) {
                return Error.ShapeMismatch;
            }

            const m = a.shape[0];
            const k = a.shape[1];
            const n = b.shape[1];

            if (b.shape[0] != k or c.shape[0] != m or c.shape[1] != n) {
                return Error.ShapeMismatch;
            }

            const a_data = a.data();
            const b_data = b.data();
            var c_data = c.data();

            // SIMD vector size
            const VECTOR_SIZE = 64 / dtypeSize;
            const Vec = @Vector(VECTOR_SIZE, T);

            var i: u32 = 0;
            while (i < m) : (i += M_BLOCK) {
                const m_end = @min(i + M_BLOCK, m);
                var j: u32 = 0;
                while (j < n) : (j += N_BLOCK) {
                    const n_end = @min(j + N_BLOCK, n);

                    // Initialize c_local
                    for (0..m_end - i) |ii| {
                        @memcpy(c_local[ii * N_BLOCK ..][0 .. n_end - j], c_data[(i + ii) * n + j ..][0 .. n_end - j]);
                    }

                    @memset(&a_local, 0);
                    @memset(&b_local, 0);

                    var p: u32 = 0;
                    while (p < k) : (p += K_BLOCK) {
                        const k_end = @min(p + K_BLOCK, k);

                        // Load A block into a_local
                        for (i..m_end) |ii| {
                            @memcpy(a_local[(ii - i) * K_BLOCK ..][0 .. k_end - p], a_data[ii * k + p ..][0 .. k_end - p]);
                        }

                        // Load B block into b_local (transposed)
                        for (p..k_end) |pp| {
                            for (j..n_end) |jj| {
                                b_local[(jj - j) * K_BLOCK + (pp - p)] = b_data[pp * n + jj];
                            }
                        }

                        // Compute using local buffers
                        for (0..m_end - i) |ii| {
                            for (0..n_end - j) |jj| {
                                var sum: Vec = @splat(0);
                                var kk: u64 = 0;

                                while (kk + VECTOR_SIZE <= k_end - p) : (kk += VECTOR_SIZE) {
                                    const block_a: Vec = a_local[ii * K_BLOCK + kk ..][0..VECTOR_SIZE].*;
                                    const block_b: Vec = b_local[jj * K_BLOCK + kk ..][0..VECTOR_SIZE].*;

                                    sum += block_a * block_b;
                                }

                                var tot = @reduce(.Add, sum);

                                for (kk..k_end - p) |kkk| {
                                    tot += a_local[ii * K_BLOCK + kkk] * b_local[jj * K_BLOCK + kkk];
                                }

                                c_local[ii * N_BLOCK + jj] += tot;
                            }
                        }
                    }

                    // Write c_local back to c_data
                    for (0..m_end - i) |ii| {
                        @memcpy(c_data[(i + ii) * n + j ..][0 .. n_end - j], c_local[ii * N_BLOCK ..][0 .. n_end - j]);
                    }
                }
            }
            return c;
        }

        pub fn cast(self: *Self, comptime newDType: dtypes.DType) (Error || Allocator.Error)!Tensor(newDType) {
            const result = try Tensor(newDType).init(self.allocator, self.shape);
            errdefer result.deinit();

            for (self.data(), result.data()) |src, *dst| {
                dst.* = switch (@typeInfo(newDType.kind)) {
                    .Float => @floatCast(src),
                    .Int => @intCast(src),
                    else => @compileError("Unsupported type for casting"),
                };
            }

            return result;
        }
    };
}

pub const dtypes = struct {
    pub const float16 = DType{ .kind = f16, .bits = 16, .name = "float16" };
    pub const float32 = DType{ .kind = f32, .bits = 32, .name = "float32" };
    pub const int16 = DType{ .kind = i16, .bits = 16, .name = "int16" };
    pub const int32 = DType{ .kind = i32, .bits = 32, .name = "int32" };
    pub const bfloat16 = DType{ .kind = BFloat16, .bits = 16, .name = "bfloat16" };
    pub const DType = struct {
        kind: type,
        bits: u8,
        name: []const u8,

        pub fn format(
            self: DType,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("dtypes.{s}", .{self.name});
        }

        pub fn castValueToKind(self: DType, value: anytype) self.kind {
            return switch (@typeInfo(self.kind)) {
                .Float => @floatCast(value),
                .Int => @intCast(value),
                .Struct => switch (self.kind) {
                    BFloat16 => BFloat16.fromF32(value),
                    else => @compileError("Unsupported type"),
                },
                else => @compileError("Unsupported type"),
            };
        }

        pub fn countBits(self: DType, value: anytype) u64 {
            return switch (@typeInfo(@TypeOf(value))) {
                .Int => @typeInfo(@TypeOf(value)).Int.bits,
                .Float => @typeInfo(@TypeOf(value)).Float.bits,
                .ComptimeInt => comptime countComptimeIntBits(value),
                .Struct => switch (self.kind) {
                    BFloat16 => 16,
                    else => @compileError("Unsupported type"),
                },
                else => @compileError("Unsupported type"),
            };
        }

        inline fn countComptimeIntBits(comptime value: anytype) u64 {
            // signed
            if (value < 0) {} else {
                comptime var count: u64 = 0;
                comptime var val: u64 = value;
                inline while (val > 0) : (count += 1) {
                    val = val >> 1;
                }
                return count;
            }
        }

        pub fn bytesFromValue(dtype: DType, bytes: *const []u8, bitCount: u64, value: dtype.kind) void {
            switch (@typeInfo(@TypeOf(value))) {
                .Int, .ComptimeInt => bytesFromInt(dtype, bytes, bitCount, value),
                .Float => bytesFromFloat(dtype, bytes, bitCount, value),
                .Struct => switch (@TypeOf(value)) {
                    BFloat16 => {
                        const val = value;
                        const as_u16 = val.bits;
                        bytes.*[0] = @as(u8, @truncate(as_u16));
                        bytes.*[1] = @as(u8, @intCast(as_u16 >> 8));
                    },
                    else => @compileError("Unsupported type"),
                },
                else => @compileError("Unsupported type"),
            }
        }

        fn bytesFromFloat(dtype: DType, bytes: *const []u8, bitCount: u64, value: dtype.kind) void {
            _ = bitCount;

            const bits = std.mem.toBytes(value);
            @memcpy(bytes.*, bits[0..]);
        }

        fn bytesFromInt(dtype: DType, bytes: *const []u8, bitCount: u64, value: dtype.kind) void {
            var i: u64 = 0;
            while (i < bitCount) : (i += 1) {
                const bit = @as(u1, @intCast((value >> @as(std.math.Log2Int(dtype.kind), @intCast(i))) & 1));
                bytes.*[i / 8] |= @as(u8, bit) << @as(u3, @truncate(@mod(i, 8)));
            }
        }

        inline fn TypeOfValue(comptime value: anytype) type {
            return switch (@typeInfo(@TypeOf(value))) {
                .ComptimeInt => comptime blk: {
                    // signed
                    if (value < 0) {} else {
                        var count = 0;
                        var val = value;
                        while (val > 0) : (count += 1) {
                            val = val >> 1;
                        }
                        break :blk std.meta.Int(.unsigned, count);
                    }
                },
                else => @TypeOf(value),
            };
        }
    };

    pub const BFloat16 = struct {
        bits: u16,

        pub inline fn fromF32(val: f32) BFloat16 {
            const as_u32: u32 = @bitCast(val);
            const bf16_bits: u16 = @truncate(as_u32 >> 16);
            return BFloat16{ .bits = bf16_bits };
        }

        pub inline fn toF32(self: *const BFloat16) f32 {
            const as_u32 = @as(u32, self.bits) << 16;
            return @bitCast(as_u32);
        }

        pub fn format(
            self: BFloat16,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("{}", .{self.toF32()});
        }

        pub inline fn max() BFloat16 {
            return BFloat16{ .bits = 0x7F7F };
        }

        pub inline fn min() BFloat16 {
            return BFloat16{ .bits = 0xFF7F };
        }
    };
};

fn determineBuffer(comptime T: type, allocator: Allocator, shape: []const u64) Allocator.Error![]u8 {
    var size: u64 = 1;

    for (shape) |dim| {
        size *= dim;
    }

    return try allocator.alloc(u8, size * @sizeOf(T) / @sizeOf(u8));
}

fn determineStrides(allocator: Allocator, shape: []const u64) Allocator.Error![]u64 {
    var strides = try allocator.alloc(u64, shape.len);

    strides[shape.len - 1] = 1;
    var i = shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = shape[i] * strides[i];
    }

    return strides;
}

// follows numpy broadcasting rules
fn broadcastShape(allocator: Allocator, shape1: []const u64, shape2: []const u64) (Error || Allocator.Error)![]u64 {
    const max_len = @max(shape1.len, shape2.len);

    var result: []u64 = try allocator.alloc(u64, max_len);
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

fn shapeFromStruct(comptime dims: anytype) [dims.len]u64 {
    if (@typeInfo(@TypeOf(dims)) != .Struct) {
        @compileError("Dimensions must be a struct");
    }

    var shape: [dims.len]u64 = undefined;
    for (dims, 0..) |dim, i| {
        if (dim < 0) {
            @compileError("Dimensions must be positive");
        }

        shape[i] = switch (@TypeOf(dim)) {
            comptime_int => dim,
            u64 => dim,
            else => @compileError("Dimensions must be of type comptime_int or u64"),
        };
    }
    return shape;
}

fn shapeFromInt(dim: anytype) [1]u64 {
    if (@typeInfo(@TypeOf(dim)) != .Int and @typeInfo(@TypeOf(dim)) != .ComptimeInt) {
        @compileError("Dimensions must be an int");
    }

    if (dim < 0) {
        @compileError("Dimensions must be positive");
    }

    return [_]u64{dim};
}

fn getShape(dims: anytype) []const u64 {
    const array = switch (@typeInfo(@TypeOf(dims))) {
        .Struct => comptime shapeFromStruct(dims),
        .Int => shapeFromInt(dims),
        .ComptimeInt => shapeFromInt(dims),
        else => @compileError("Unsupported type. Expected struct or int"),
    };

    return array[0..];
}

fn shapeOfSlice(comptime T: type, allocator: Allocator, dims: anytype) Allocator.Error![]u64 {
    const dim: u64 = struct {
        fn countDims(slice: anytype, acc: u64) u64 {
            if (@typeInfo(@TypeOf(slice)) == .Pointer) {
                if (slice.len == 0) {
                    return acc + 1;
                } else {
                    return countDims(slice[0], acc + 1);
                }
            } else if (@TypeOf(slice) == T) {
                return acc;
            } else {
                @compileError("Unsupported type");
            }
        }
    }.countDims(dims, 0);

    const shape: []u64 = try allocator.alloc(u64, dim);

    struct {
        fn setShape(_shape: []u64, _slice: anytype, idx: u64) void {
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
