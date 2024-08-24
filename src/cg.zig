const std = @import("std");
const FieldType = std.meta.FieldType;

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
        Sum: void,
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
    view: *const AnyView,
    dtype: dtypes.DataType,

    const Self = @This();

    pub fn init(
        comptime op: Operations,
        arg: FieldType(Operations.Arg, op),
        input: FieldType(Operations.Input, op),
        comptime view: *const AnyView,
        comptime dtype: dtypes.DataType,
    ) Self {
        return Self{
            .op = op,
            .arg = @unionInit(Operations.Arg, @tagName(op), arg),
            .input = @unionInit(Operations.Input, @tagName(op), input),
            .view = view,
            .dtype = dtype,
        };
    }
};

pub const BufferID = u32;

pub const MemoryManager = struct {
    const Self = @This();

    buffers: std.AutoHashMapUnmanaged(BufferID, RuntimeBuffer),

    pub fn init() Self {
        return Self{
            .buffers = .{},
        };
    }

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        self.buffers.deinit(allocator);
    }

    pub fn create_id(comptime T: type) BufferID {
        comptime {
            const type_name = @typeName(T);

            const index = blk: {
                const N = type_name.len;
                for (0..N) |i| {
                    if (type_name[N - 1 - i] == '_') {
                        break :blk N - 1 - i;
                    }
                }
                @compileError("Pass in opaque {} to MemoryManager.create_id");
            };

            var value: BufferID = 0;
            for (type_name[index + 1 ..]) |c| {
                value = 10 * value + (@as(BufferID, c) - '0');
            }
            return value;
        }
    }

    pub fn set_buffer(self: *Self, allocator: std.mem.Allocator, id: BufferID, buffer: RuntimeBuffer) !void {
        try self.buffers.putNoClobber(allocator, id, buffer);
    }
};

pub const RuntimeBuffer = struct {
    ptr: []const u8,
    len: u32,
    dtype: dtypes.DataType,
    shape: []const u32,
};

pub const Numpy = struct {
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !RuntimeBuffer {
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var reader = file.reader();

        var magic: [6]u8 = undefined;
        try reader.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "\x93NUMPY")) {
            return error.InvalidNumpyFormat;
        }

        const major_version = try reader.readByte();
        const minor_version = try reader.readByte();
        if (major_version != 1 or minor_version != 0) {
            return error.UnsupportedNumpyVersion;
        }

        const header_len = try reader.readInt(u16, std.builtin.Endian.little);

        const header = try allocator.alloc(u8, header_len);
        defer allocator.free(header);
        try reader.readNoEof(header);

        var dtype: []const u8 = undefined;
        var fortran_order: bool = false;
        var shape: []u32 = undefined;

        var it = std.mem.splitScalar(u8, header, ',');
        while (it.next()) |item| {
            var kv = std.mem.splitScalar(u8, item, ':');

            if (kv.next()) |key| {
                if (std.mem.indexOf(u8, key, "descr") != null) {
                    dtype = std.mem.trim(u8, kv.next() orelse return error.InvalidHeader, " '");
                } else if (std.mem.indexOf(u8, key, "fortran_order") != null) {
                    fortran_order = if (std.mem.indexOf(u8, kv.next() orelse return error.InvalidHeader, "True") != null) true else false;
                } else if (std.mem.indexOf(u8, key, "shape") != null) {
                    const shape_str = std.mem.trim(u8, kv.next() orelse return error.InvalidHeader, " ()");
                    var shape_it = std.mem.splitScalar(u8, shape_str, ' ');
                    var shape_list = std.ArrayList(u32).init(allocator);
                    defer shape_list.deinit();
                    while (shape_it.next()) |dim| {
                        try shape_list.append(try std.fmt.parseInt(u32, dim, 10));
                    }
                    shape = try shape_list.toOwnedSlice();
                }
            }
        }

        const datatype = dtypes.FromNumpy(dtype) orelse return error.Invalid;

        var size: u32 = 1;
        for (shape) |dim| {
            size *= dim;
        }

        const data: []u8 = try allocator.alloc(u8, size * datatype.bits / 8);

        try reader.readNoEof(data);

        return RuntimeBuffer{
            .ptr = data,
            .len = size,
            .dtype = datatype,
            .shape = shape,
        };
    }
};

pub const AnyView = extern struct {
    shape: [*]u32,
    strides: [*]u32,
    offset: u32,
    mask: ?[*]u32,
    contiguous: bool,
    rank: u32,
};

pub fn View(comptime shape: []const u32) type {
    comptime {
        const rank: u32 = @intCast(shape.len);

        const strides: [rank]u32 = blk: {
            var strides: [rank]u32 = undefined;
            strides[rank - 1] = 1;
            var i = rank - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = shape[i] * strides[i];
            }
            break :blk strides;
        };

        return extern struct {
            const Self = @This();

            shape: *const [rank]u32,
            strides: *const [rank]u32,
            offset: u32,
            mask: ?*const [rank]u32,
            contiguous: bool,
            rank: u32,

            pub fn init() Self {
                return Self{
                    .shape = shape[0..],
                    .strides = strides[0..],
                    .offset = 0,
                    .mask = null,
                    .contiguous = true,
                    .rank = rank,
                };
            }

            pub fn format(
                self: Self,
                comptime fmt: []const u8,
                options: std.fmt.FormatOptions,
                writer: anytype,
            ) !void {
                _ = options;
                _ = fmt;
                try writer.print("View(", .{});

                try writer.print("shape=(", .{});
                for (self.shape, 0..self.rank) |dim, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{dim});
                }
                try writer.print("),", .{});

                try writer.print(" strides=(", .{});
                for (self.strides, 0..self.rank) |stride, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{stride});
                }
                try writer.print("),", .{});

                try writer.print(" offset={},", .{self.offset});

                try writer.print(" mask=", .{});
                if (self.mask) |mask| {
                    try writer.print("(", .{});
                    for (mask, 0..self.rank) |dim, i| {
                        if (i > 0) try writer.print(", ", .{});
                        try writer.print("{}", .{dim});
                    }
                    try writer.print("),", .{});
                } else {
                    try writer.print("None,", .{});
                }

                try writer.print(" contiguous={}", .{self.contiguous});

                try writer.print(")", .{});
            }
        };
    }
}

pub const Shape = extern struct {
    const Self = @This();

    dims: [*]const u32,
    rank: u32,
    size: u32,
    strides: [*]const u32,

    pub fn init(allocator: std.mem.Allocator, dims: []const u32) !Self {
        const size = blk: {
            var count: u32 = 1;
            for (0..dims.len, dims) |_, dim| {
                count *= dim;
            }
            break :blk count;
        };
        const rank: u32 = @intCast(dims.len);

        var strides = try allocator.alloc(u32, rank);

        strides[rank - 1] = 1;
        var i = rank - 1;
        while (i > 0) : (i -= 1) {
            strides[i - 1] = dims[i] * strides[i];
        }

        return Self{
            .dims = @ptrCast(dims),
            .rank = rank,
            .size = size,
            .strides = @ptrCast(&strides),
        };
    }

    pub fn equal(self: Self, other: Self) bool {
        return std.mem.eql(u32, self.dims, other.dims);
    }

    pub fn print(self: Self) void {
        std.debug.print("Shape(", .{});
        for (0..self.rank, self.dims) |_, dim| {
            std.debug.print("{}, ", .{dim});
        }
        std.debug.print(")\n", .{});
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;
        try writer.print("[", .{});
        for (self.dims, 0..self.rank) |dim, i| {
            if (i > 0) try writer.print(", ", .{});
            try writer.print("{}", .{dim});
        }
    }
};

pub fn Tensor(comptime datatype: dtypes.DataType, comptime dims: [*]const u32) type {
    const shape_len = dims.len;

    return extern struct {
        const Self = @This();

        pub const dtype: dtypes.DataType = datatype;
        pub const shape: [shape_len]u32 = dims;
        pub const strides: [shape_len]u32 = undefined;
        node: ?*Node,

        pub fn init() Self {
            return Self{
                .node = null,
            };
        }
    };
}

pub const ExecutionStep = struct {
    node: *const Node,
    dependencies_completed: u32,
};

pub const ExecutionPlan = struct {
    const Self = @This();

    steps: []ExecutionStep,
    estimated_memory: u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.steps);
    }
};

pub const Scheduler = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    memory_manager: MemoryManager,

    pub fn init(allocator: std.mem.Allocator, memory_manager: MemoryManager) Self {
        return Self{
            .allocator = allocator,
            .memory_manager = memory_manager,
        };
    }

    pub fn deinit(self: Self) void {
        defer self.visited.deinit();
    }

    pub fn run(self: Self, comptime node: *const Node) !ExecutionPlan {
        var visited = std.AutoHashMap(*const Node, void).init(self.allocator);
        defer visited.deinit();

        var order = std.ArrayList(*const Node).init(self.allocator);
        defer order.deinit();

        try Self.topologicalSort(node, &visited, &order);

        var steps = try self.allocator.alloc(ExecutionStep, order.items.len);
        errdefer self.allocator.free(steps);

        var estimated_memory: u32 = 0;

        for (order.items, 0..) |n, i| {
            steps[i] = ExecutionStep{
                .node = n,
                .dependencies_completed = 0,
            };
            estimated_memory += estimate_memory_usage(n);
        }

        return ExecutionPlan{
            .steps = steps,
            .estimated_memory = estimated_memory,
            .allocator = self.allocator,
        };
    }

    fn topologicalSort(
        comptime node: *const Node,
        visited: *std.AutoHashMap(*const Node, void),
        order: *std.ArrayList(*const Node),
    ) !void {
        if (visited.contains(node)) {
            return;
        }

        try visited.put(node, {});

        const inputs = @field(node.input, @tagName(node.op));

        if (@typeInfo(@TypeOf(inputs)) == .Array) {
            inline for (inputs) |input| {
                try topologicalSort(input, visited, order);
            }
        }

        try order.append(node);
    }

    fn estimate_memory_usage(node: *const Node) u32 {
        const shape = node.view.shape;
        const rank = node.view.rank;

        const size = blk: {
            var count: u32 = 1;
            for (0..rank, shape) |_, dim| {
                count *= dim;
            }
            break :blk count;
        };

        return size * node.dtype.bits / 8;
    }
};

pub const tensor = extern struct {
    const Self = @This();

    dtype: dtypes.DataType,
    shape: Shape,
    node: ?*Node,
    data: ?[*]u8,

    fn AsTensor(comptime self: *const Self) type {
        return Tensor(self.dtype, self.shape);
    }

    pub fn to(comptime self: *const Self) Self.AsTensor(self.*) {
        return @ptrCast(self);
    }

    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;

        const view = if (self.node) |node| node.view else return error.Invalid;

        const formatData = struct {
            fn print(wrt: anytype, shape: Shape, d: [](i64), currentShapeIndex: u32, offset: u32) !void {
                for (0..currentShapeIndex + 1) |_| {
                    try wrt.print("\t", .{});
                }

                try wrt.print("[", .{});

                // last shape
                if (currentShapeIndex == shape.rank - 1) {
                    for (0..shape.dims[currentShapeIndex]) |idx| {
                        if (idx > 0) {
                            try wrt.print(", ", .{});
                        }
                        try wrt.print("{}", .{d[idx + offset]});
                    }
                } else {
                    try wrt.print("\n", .{});

                    for (0..shape.dims[currentShapeIndex]) |idx| {
                        if (idx > 0) {
                            try wrt.print(",\n", .{});
                        }
                        try print(wrt, shape, d, currentShapeIndex + 1, offset + @as(u32, @intCast(idx)) * shape.strides[currentShapeIndex]);
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

        try writer.print("\ttype: {},\n", .{self.dtype});

        try writer.print("\tshape: {},\n", .{view});

        try writer.print("\tlength: {},\n", .{view.rank});

        try writer.print("\tdata:\n", .{});
        try formatData(writer, view, @constCast(&[_]i64{ 1, 2, 3 })[0..], 0, 0);
        try writer.print("\n", .{});

        try writer.print(")", .{});
    }
};

pub const DTypeNames = enum(u8) {
    Int32,
    Int64,
    Float32,
    Float64,
};

pub const dtypes = struct {
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
};
