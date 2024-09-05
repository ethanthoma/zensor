const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;

pub const Error = error{
    ShapeMismatch,
    CannotBroadcast,
    WrongType,
};

pub var prng = std.Random.DefaultPrng.init(20240807);

pub fn manualSeed(seed: u64) void {
    prng = std.Random.DefaultPrng.init(seed);
}

pub fn Tensor(comptime dtype: dtypes.DType) type {
    const T = dtype.kind;
    return struct {
        const Self = @This();

        buffer: []u8,
        shape: []u32,
        strides: []u32,
        allocator: Allocator,

        pub usingnamespace Creation(dtype);
        pub usingnamespace Movement(dtype);
        pub usingnamespace Ops(dtype);

        fn init(allocator: Allocator, shape: []const u32) Allocator.Error!Self {
            return Self{
                .buffer = try determineBuffer(T, allocator, shape),
                .shape = try allocator.dupe(u32, shape),
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

            const shape = try self.allocator.dupe(u32, self.shape);
            errdefer self.allocator.free(shape);

            const strides = try self.allocator.dupe(u32, self.strides);

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

        pub fn at(self: Self, index: u32) T {
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
                fn print(wrt: anytype, strides: @TypeOf(self.strides), shape: @TypeOf(self.shape), d: []T, currentShapeIndex: u32, offset: u32) !void {
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
                            try print(wrt, strides, shape, d, currentShapeIndex + 1, offset + @as(u32, @intCast(idx)) * strides[currentShapeIndex]);
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

        fn broadcastToShape(self: *Self, targetShape: []const u32) Allocator.Error!void {
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

            const shape = try self.allocator.dupe(u32, targetShape);
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
            const dimLength = dimLengthFromType(dims);
            const shape: []u32 = try allocator.alloc(u32, dimLength);
            getShape(shape, dims);

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

        pub fn arange(allocator: Allocator, start: u32, stop: u32) (Error || Allocator.Error)!Self {
            const self = try Self.empty(allocator, stop - start);

            const convertToT = struct {
                pub fn func(from: usize) dtype.kind {
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
                pub fn copyData(_slice: anytype, _data: []dtype.kind, _strides: []u32, offset: u32, depth: u32) void {
                    switch (@TypeOf(_slice)) {
                        []const dtype.kind => {
                            @memcpy(_data[offset .. offset + _slice.len], _slice);
                        },
                        else => {
                            for (_slice, 0..) |_innerSlice, idx| {
                                copyData(_innerSlice, _data, _strides, offset + @as(u32, @intCast(idx)) * _strides[depth], depth + 1);
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

        pub fn rand(allocator: Allocator, dims: anytype) (Error || Allocator.Error)!Self {
            const self = try Self.empty(allocator, dims);
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

        pub fn randInt(allocator: Allocator, dims: anytype, low: dtype.kind, high: dtype.kind) (Error || Allocator.Error)!Self {
            if (@typeInfo(dtype.kind) != .Int) {
                return Error.WrongType;
            }

            const self = try Self.empty(allocator, dims);
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
        pub fn reshape(self: *Self, dims: anytype) (Error || Allocator.Error)!void {
            const dimLength = dimLengthFromType(dims);
            const shape: []u32 = try self.allocator.alloc(u32, dimLength);
            getShape(shape, dims);
            errdefer self.allocator.free(shape);

            var length: u32 = 1;
            for (shape) |dim| {
                length *= dim;
            }

            if (length != self.data().len) {
                return Error.ShapeMismatch;
            }

            self.allocator.free(self.shape);
            self.allocator.free(self.strides);

            self.shape = try self.allocator.dupe(u32, shape);
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

        const kernels = struct {
            const VECTOR_SIZE = 16;
            const Vec = @Vector(VECTOR_SIZE, T);

            const L1_CACHE_SIZE = 1024 * 64;

            const KERNEL_SIZE_M = 8;
            const KERNEL_SIZE_N = 16;
            const KERNEL_SIZE_K = 128;

            pub inline fn matmul_dot_inner(
                comptime regsA: u32,
                comptime regsB: u32,
                comptime k: u32,
                a: []const T,
                lda: u32,
                b: []const T,
                ldb: u32,
                c: []T,
                ldc: u32,
            ) void {
                var csum: [regsA][regsB]Vec = .{.{@as(Vec, @splat(0))} ** regsB} ** regsA;

                for (0..k) |p| {
                    for (0..regsB) |bi| {
                        const bb: Vec = b[p * ldb + bi * VECTOR_SIZE ..][0..VECTOR_SIZE].*;
                        for (0..regsA) |ai| {
                            const aa: Vec = @splat(a[ai * lda + p]);
                            csum[ai][bi] += aa * bb;
                        }
                    }
                }

                for (0..regsA) |ai| {
                    for (0..regsB) |bi| {
                        var cc: Vec = c[ai * ldc + bi * VECTOR_SIZE ..][0..VECTOR_SIZE].*;
                        cc += csum[ai][bi];
                        @memcpy(c[ai * ldc + bi * VECTOR_SIZE ..], &@as([VECTOR_SIZE]T, cc));
                    }
                }
            }

            pub inline fn matmul_scaler(
                a_data: []const T,
                lda: u32,
                b_data: []const T,
                ldb: u32,
                c_data: []T,
                ldc: u32,
                m_start: u32,
                m_end: u32,
                n_start: u32,
                n_end: u32,
                k_start: u32,
                k_end: u32,
            ) void {
                for (m_start..m_end) |i| {
                    for (n_start..n_end) |j| {
                        var sum: T = 0;
                        @prefetch(a_data[i * lda ..], .{});
                        for (k_start..k_end) |p| {
                            sum += a_data[i * lda + p] * b_data[p * ldb + j];
                        }
                        c_data[i * ldc + j] += sum;
                    }
                }
            }
        };

        pub fn matmuladd(c: *Self, a: *const Self, b: *const Self) (Error || Allocator.Error)!*Self {
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

            const KERNEL_SIZE_M = kernels.KERNEL_SIZE_M;
            const KERNEL_SIZE_N = kernels.KERNEL_SIZE_N;
            const KERNEL_SIZE_K = kernels.KERNEL_SIZE_K;

            const m_tiled: u32 = (m / KERNEL_SIZE_M) * KERNEL_SIZE_M;
            const n_tiled: u32 = (n / KERNEL_SIZE_N) * KERNEL_SIZE_N;
            const k_tiled: u32 = (k / KERNEL_SIZE_K) * KERNEL_SIZE_K;

            var i: u32 = 0;
            while (i < m_tiled) : (i += KERNEL_SIZE_M) {
                var j: u32 = 0;
                while (j < n_tiled) : (j += KERNEL_SIZE_N) {
                    var p: u32 = 0;
                    while (p < k_tiled) : (p += KERNEL_SIZE_K) {
                        kernels.matmul_dot_inner(
                            KERNEL_SIZE_M,
                            KERNEL_SIZE_N / kernels.VECTOR_SIZE,
                            KERNEL_SIZE_K,
                            a_data[i * k + p ..],
                            k,
                            b_data[p * n + j ..],
                            n,
                            c_data[i * n + j ..],
                            n,
                        );
                    }
                }
            }

            if (m_tiled < m) {
                kernels.matmul_scaler(a_data, k, b_data, n, c_data, n, m_tiled, m, 0, n, 0, k);
            }

            if (n_tiled < n) {
                kernels.matmul_scaler(a_data, k, b_data, n, c_data, n, 0, m_tiled, n_tiled, n, 0, k);
            }

            if (k_tiled < k) {
                kernels.matmul_scaler(a_data, k, b_data, n, c_data, n, 0, m_tiled, 0, n_tiled, k_tiled, k);
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

        pub fn countBits(self: DType, value: anytype) u32 {
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

        inline fn countComptimeIntBits(comptime value: anytype) u32 {
            // signed
            if (value < 0) {} else {
                comptime var count: u32 = 0;
                comptime var val: u32 = value;
                inline while (val > 0) : (count += 1) {
                    val = val >> 1;
                }
                return count;
            }
        }

        pub fn bytesFromValue(dtype: DType, bytes: *const []u8, bitCount: u32, value: dtype.kind) void {
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

        fn bytesFromFloat(dtype: DType, bytes: *const []u8, bitCount: u32, value: dtype.kind) void {
            _ = bitCount;

            const bits = std.mem.toBytes(value);
            @memcpy(bytes.*, bits[0..]);
        }

        fn bytesFromInt(dtype: DType, bytes: *const []u8, bitCount: u32, value: dtype.kind) void {
            var i: u32 = 0;
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

fn determineBuffer(comptime T: type, allocator: Allocator, shape: []const u32) Allocator.Error![]u8 {
    var size: u32 = 1;

    for (shape) |dim| {
        size *= dim;
    }

    return try allocator.alloc(u8, size * @sizeOf(T) / @sizeOf(u8));
}

fn determineStrides(allocator: Allocator, shape: []const u32) Allocator.Error![]u32 {
    var strides = try allocator.alloc(u32, shape.len);

    strides[shape.len - 1] = 1;
    var i = shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = shape[i] * strides[i];
    }

    return strides;
}

// follows numpy broadcasting rules
fn broadcastShape(allocator: Allocator, shape1: []const u32, shape2: []const u32) (Error || Allocator.Error)![]u32 {
    const max_len = @max(shape1.len, shape2.len);

    var result: []u32 = try allocator.alloc(u32, max_len);
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

fn dimLengthFromType(dims: anytype) u32 {
    return switch (@typeInfo(@TypeOf(dims))) {
        .Struct => comptime dims.len,
        .Int => 1,
        .ComptimeInt => comptime 1,
        else => @compileError("Dimensions must be an int or struct"),
    };
}

fn shapeFromStruct(shape: []u32, comptime dims: anytype) void {
    comptime var shape_arr: [dims.len]u32 = undefined;

    comptime {
        for (dims, 0..) |dim, i| {
            if (dim < 0) {
                @compileError("Dimensions must be positive");
            }

            shape_arr[i] = switch (@typeInfo(@TypeOf(dim))) {
                .Int, .ComptimeInt => @intCast(dim),
                else => @compileError("Dimensions must be of type comptime_int or u32"),
            };
        }
    }

    for (0..shape.len) |i| {
        shape[i] = shape_arr[i];
    }
}

fn getShape(shape: []u32, dims: anytype) void {
    switch (@typeInfo(@TypeOf(dims))) {
        .Struct => shapeFromStruct(shape, dims),
        .Int, .ComptimeInt => shape[0] = @intCast(dims),
        else => @compileError("Unsupported type. Expected struct or int"),
    }
}

fn shapeOfSlice(comptime T: type, allocator: Allocator, dims: anytype) Allocator.Error![]u32 {
    const dim: u32 = struct {
        fn countDims(slice: anytype, acc: u32) u32 {
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

    const shape: []u32 = try allocator.alloc(u32, dim);

    struct {
        fn setShape(_shape: []u32, _slice: anytype, idx: u32) void {
            if (@typeInfo(@TypeOf(_slice)) == .Pointer) {
                _shape[idx] = @intCast(_slice.len);
                if (_slice.len > 0) {
                    setShape(_shape, _slice[0], idx + 1);
                }
            }
        }
    }.setShape(shape, dims, 0);

    return shape;
}

pub const TensorType = enum(u8) {
    Float16,
    Float32,

    fn getType(datatype: TensorType) type {
        return comptime switch (datatype) {
            TensorType.Float16 => f16,
            TensorType.Float32 => f32,
        };
    }
};

pub const DynamicTensor = extern struct {
    const Self = @This();

    datatype: TensorType,
    data: [*]u8,

    fn Narrow(comptime self: Self) type {
        return Tensor(self.datatype);
    }

    pub fn narrow(comptime self: *const Self) *const Narrow(self.*) {
        return @ptrCast(self);
    }

    pub fn init(allocator: Allocator, comptime datatype: TensorType, data: []u8) Self {
        return Self{ .datatype = datatype, .data = data, .allocator = allocator };
    }

    pub fn deinit(self: Self) void {
        self.allocator.free(self.data);
    }

    pub fn full(allocator: Allocator, comptime datatype: TensorType, length: u32, value: anytype) (Error || Allocator.Error)!Self {
        const Type = comptime datatype.getType();
        const data = try allocator.alloc(u8, length * @sizeOf(Type));
        @memset(data, value);
        return Self.init(allocator, datatype, data);
    }

    pub fn printData(self: Self) void {
        const Type = comptime self.datatype.getType();
        const data = mem.bytesAsSlice(Type, self.data);
        for (data) |elem| {
            std.debug.print("{}, ", .{elem});
        }
    }
};
