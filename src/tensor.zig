const std = @import("std");

const Compiler = @import("Compiler.zig");
const ast = Compiler.ast;
const Scheduler = Compiler.Scheduler;
const IRGenerator = Compiler.IRGenerator;

const dtypes = @import("dtypes.zig");
const view = @import("view.zig");
const RuntimeBuffer = @import("RuntimeBuffer.zig");

pub fn Tensor(comptime datatype: dtypes.DType, comptime shape: anytype) type {
    return struct {
        const Self = @This();

        const anyview = view.View(array_init_shape(shape)).init().as_any_view();
        const dtype = datatype;

        index: Compiler.NodeIndex,
        compiler: *Compiler,

        pub fn realize(self: Self) !*RuntimeBuffer {
            self.compiler.mark_for_scheduling(self.index);

            return self.compiler.run(self.index);
        }

        fn full_node(allocator: std.mem.Allocator, comptime value: anytype) !ast.Node {
            return ast.Node.init(
                .Const,
                .{ .value = try std.fmt.allocPrint(allocator, "{}", .{value}) },
                {},
                anyview,
                dtype,
            );
        }

        pub fn full(compiler: *Compiler, comptime value: anytype) !Self {
            const node = try full_node(compiler.allocator, value);
            const index = try compiler.node_manager.add(node);
            return Self{ .index = index, .compiler = compiler };
        }

        fn load_node(buffer: *RuntimeBuffer) !ast.Node {
            return ast.Node.init(
                .Load,
                .{ .buffer = buffer },
                {},
                anyview,
                dtype,
            );
        }

        pub fn from_numpy(compiler: *Compiler, comptime filename: []const u8) !Self {
            const allocator = compiler.allocator;

            const buffer = try allocator.create(RuntimeBuffer);
            errdefer allocator.destroy(buffer);
            buffer.* = try RuntimeBuffer.Numpy.load(allocator, filename);

            const node = try load_node(buffer);
            const index = try compiler.node_manager.add(node);

            try compiler.register_buffer(index, buffer);

            return Self{ .index = index, .compiler = compiler };
        }

        pub fn mul(self: Self, other: Self) !Self {
            return binary_op(.Mul, self, other);
        }

        pub fn sum(self: Self, comptime dim: u32) !Tensor(
            Self.dtype,
            anyview.as_view().reduce(dim).shape,
        ) {
            return reduce_op(.Sum, self, dim);
        }

        fn binary_op(comptime op: ast.Operation, lhs: Self, rhs: Self) !Self {
            if (comptime op.AsOperationType() != .Binary) {
                @compileError(std.fmt.comptimePrint("Expected binary op, got {}", .{op.AsOperationType()}));
            }

            const node_lhs = lhs.compiler.node_manager.get(lhs.index).?;
            const node_rhs = rhs.compiler.node_manager.get(rhs.index).?;

            const node = ast.Node.init(
                op,
                {},
                .{ node_lhs, node_rhs },
                anyview,
                dtype,
            );
            const index = try lhs.compiler.node_manager.add(node);

            return Self{ .index = index, .compiler = lhs.compiler };
        }

        fn reduce_op(comptime op: ast.Operation, self: Self, comptime dim: u32) !Tensor(
            Self.dtype,
            anyview.as_view().reduce(dim).shape,
        ) {
            if (comptime op.AsOperationType() != .Reduce) {
                @compileError(std.fmt.comptimePrint("Expected reduce op, got {}", .{op.AsOperationType()}));
            }

            const reduce_view = comptime anyview.as_view().reduce(dim);

            const node = ast.Node.init(
                op,
                .{ .dim = dim },
                .{self.compiler.node_manager.get(self.index).?},
                comptime reduce_view.as_any_view(),
                dtype,
            );

            const index = try self.compiler.node_manager.add(node);

            return Tensor(Self.dtype, reduce_view.shape){
                .index = index,
                .compiler = self.compiler,
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

            const buffer = try self.compiler.run(self.index);

            try writer.print("Tensor(\n", .{});

            try writer.print("\ttype: {},\n", .{dtype});

            try writer.print("\tshape: [", .{});
            for (anyview.shape, 0..anyview.rank) |dim, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{}", .{dim});
            }
            try writer.print("],\n", .{});

            try writer.print("\tlength: {},\n", .{anyview.size});

            try writer.writeAll("\tdata: [");
            for (0..anyview.size) |i| {
                const value_buffer = buffer.get(@intCast(i)).?;
                const value = std.mem.bytesToValue(dtype.ToBuiltin(), value_buffer);
                try writer.print("{}, ", .{value});
            }
            try writer.writeAll("]\n");

            try writer.print(")", .{});
        }
    };
}

/// creates a shape array []const u32 from another slice, array, pointer to array, or tuple
fn array_init_shape(comptime tuple: anytype) []const u32 {
    comptime {
        switch (@typeInfo(@TypeOf(tuple))) {
            .@"struct", .@"union", .@"enum" => {
                const fields = std.meta.fields(@TypeOf(tuple));

                var shape: [fields.len]u32 = undefined;

                for (fields, 0..) |field, i| {
                    switch (@typeInfo(field.type)) {
                        .int, .comptime_int => {
                            const value = @field(tuple, field.name);

                            if (value <= 0) {
                                @compileError(std.fmt.comptimePrint(
                                    "Shape dims must be greater than 0." ++
                                        "  You passed in {} at index {}",
                                    .{ value, i },
                                ));
                            }

                            shape[i] = value;
                        },
                        else => @compileError(std.fmt.comptimePrint(
                            "Shape dims must be integers." ++
                                "  You passed in type {} at index {}",
                            .{ field.type, i },
                        )),
                    }
                }

                const final_shape = shape;
                return &final_shape;
            },
            .pointer => |info| {
                if (info.size == .slice) {
                    switch (@typeInfo(info.child)) {
                        .int, .comptime_int => {
                            var shape: [info.child.len]u32 = undefined;

                            for (tuple, 0..) |elem, i| {
                                shape[i] = elem;
                            }

                            const final_shape = shape;
                            return &final_shape;
                        },
                        else => @compileError(std.fmt.comptimePrint(
                            "Shape dims must be integers." ++
                                "  You passed in type {}",
                            .{info.child},
                        )),
                    }
                } else if (info.size == .one) {
                    return array_init_shape(tuple.*);
                } else {
                    @compileError(std.fmt.comptimePrint("Shape dims must be a tuple, slice, or array." ++
                        "  You passed in type {}", .{@TypeOf(tuple)}));
                }
            },
            .array => |info| {
                switch (@typeInfo(info.child)) {
                    .int, .comptime_int => {
                        var shape: [info.len]u32 = undefined;

                        for (tuple, 0..) |elem, i| {
                            shape[i] = elem;
                        }

                        const final_shape = shape;
                        return &final_shape;
                    },
                    else => @compileError(std.fmt.comptimePrint(
                        "Shape dims must be integers." ++
                            "  You passed in type {}",
                        .{info.child},
                    )),
                }
            },
            else => {
                @compileError(std.fmt.comptimePrint("Shape dims must be a tuple, slice, or array." ++
                    "  You passed in type {}", .{@TypeOf(tuple)}));
            },
        }
    }
}

const Metadata = struct {
    dtype: dtypes.DType,
    anyview: view.AnyView,
    node: *const ast.Node,
};

fn to_meta_data(T: type) Metadata {
    verify_tensor(T);
    return comptime Metadata{
        .dtype = T.dtype,
        .anyview = T.anyview,
        .node = T.node,
    };
}

// TODO: should probably be moved to view.zig
fn binary_verify_shapes(comptime lhs: view.AnyView, comptime rhs: view.AnyView) void {
    comptime {
        const shape_lhs = lhs.as_view().shape;
        const shape_rhs = rhs.as_view().shape;

        if (shape_lhs.len != shape_rhs.len) {
            @compileError("Shapes must have the same length");
        }

        // TODO: change for broadcasting later
        for (shape_lhs, shape_rhs, 0..) |l, r, i| {
            if (l != r) {
                @compileError(std.fmt.comptimePrint(
                    "Mismatching dim at position {}:\n" ++
                        "{} != {}",
                    .{ i, l, r },
                ));
            }
        }
    }
}

/// Verifies that the other object in tensor ops are tensor objects.
/// The ast.Node is known at comptime but I couldn't find a way to abuse pointer
/// casting to get runtime scheduler and comptime node access to be unified.
/// This is based on the hashmap Context verifying function in std.
fn verify_tensor(Context: type) void {
    const prefix = ".  ";
    const context_error = "Must be a Operations type with dtype, view, and node";
    const cannot_use = &struct {
        fn f(T: type) []const u8 {
            return "Cannot use type of " ++ @typeName(T);
        }
    }.f;
    const print = struct {
        fn f(comptime args: anytype) []const u8 {
            const ArgsType = @TypeOf(args);
            const fields = std.meta.fields(ArgsType);

            comptime var str: []const u8 = "";

            inline for (fields, 0..) |field, i| {
                const arg = @field(args, field.name);
                str = str ++ arg;
                if (!std.mem.eql(u8, str, "") and i < fields.len - 1) {
                    str = str ++ prefix;
                }
            }

            const finalized_str: []const u8 = str;
            return finalized_str;
        }
    }.f;

    comptime {
        var errors: []const u8 = "";

        switch (@typeInfo(Context)) {
            .Struct, .Union, .Enum => {},
            .Opaque => {
                errors = print(.{
                    errors,
                    context_error,
                    cannot_use(Context),
                    "Use a single pointer instead of an opaque",
                });
            },
            .Pointer => |ptr| {
                if (ptr.size != .One) {
                    errors = print(.{
                        errors,
                        context_error,
                        cannot_use(Context),
                        "Must be a single pointer",
                    });
                }
                Context = ptr.child;
                switch (@typeInfo(Context)) {
                    .Struct, .Union, .Enum, .Opaque => {},
                    else => {
                        errors = print(.{
                            errors,
                            context_error,
                            cannot_use(Context),
                        });
                    },
                }
            },
            else => {
                errors = print(.{
                    errors,
                    context_error,
                    cannot_use(Context),
                });
            },
        }

        if (errors.len != 0) {
            @compileError("Problems found with Operations type:\n" ++ errors);
        }

        const error_msgs = struct {
            const dtype_type_name = @typeName(dtypes.DataType);
            const invalid_dtype_type =
                "The dtype declaration must be " ++
                dtype_type_name ++
                " but is actually " ++
                @typeName(@TypeOf(Context.dtype));
            const not_pub_dtype =
                "Operations type must declare a pub declaration with type " ++
                dtype_type_name;
            const anyview_type_name = @typeName(view.AnyView);
            const invalid_anyview_type =
                "The anyview declaration must be " ++
                anyview_type_name ++
                " but is actually " ++
                @typeName(@TypeOf(Context.anyview));
            const not_pub_anyview =
                "Operations type must declare a pub declaration with type " ++
                anyview_type_name;
            const node_type_name = @typeName(*const ast.Node);
            const invalid_node_type =
                "The node declaration must be " ++
                node_type_name ++
                " but is actually " ++
                @typeName(@TypeOf(Context.node));
            const not_pub_node =
                "Operations type must declare a pub declaration with type " ++
                node_type_name;
        };

        if (@hasDecl(Context, "dtype")) {
            if (@TypeOf(Context.dtype) != dtypes.DType) {
                errors = errors ++ "\n" ++ error_msgs.invalid_dtype_type;
            }
        } else {
            errors = errors ++ "\n" ++ error_msgs.not_pub_dtype;
        }

        if (@hasDecl(Context, "anyview")) {
            if (@TypeOf(Context.anyview) != view.AnyView) {
                errors = errors ++ "\n" ++ error_msgs.invalid_anyview_type;
            }
        } else {
            errors = errors ++ "\n" ++ error_msgs.not_pub_anyview;
        }

        if (@hasDecl(Context, "node")) {
            if (@TypeOf(Context.node) != *const ast.Node) {
                errors = errors ++ "\n" ++ error_msgs.invalid_node_type;
            }
        } else {
            errors = errors ++ "\n" ++ error_msgs.not_pub_node;
        }

        if (errors.len != 0) {
            @compileError("Problems found with Operations type:\n" ++ errors);
        }
    }
}
