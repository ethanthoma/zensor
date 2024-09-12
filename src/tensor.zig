const std = @import("std");

const Compiler = @import("Compiler.zig");
const ast = Compiler.ast;
const Scheduler = Compiler.Scheduler;
const IRGenerator = Compiler.IRGenerator;

const dtypes = @import("dtypes.zig");
const view = @import("view.zig");
const RuntimeBuffer = @import("RuntimeBuffer.zig");

const CPU = @import("backend/CPU.zig");

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

        fn mul_node(lhs: *ast.Node, rhs: *ast.Node) ast.Node {
            return ast.Node.init(
                .Mul,
                {},
                .{ lhs, rhs },
                anyview,
                dtype,
            );
        }

        pub fn mul(self: Self, other: Self) !Self {
            const lhs = self.compiler.node_manager.get(self.index).?;
            const rhs = self.compiler.node_manager.get(other.index).?;

            const node = mul_node(lhs, rhs);
            const index = try self.compiler.node_manager.add(node);

            return Self{ .index = index, .compiler = self.compiler };
        }

        fn sum_node(self: *ast.Node, comptime dim: u32) ast.Node {
            return ast.Node.init(
                .Sum,
                .{ .dim = dim },
                .{self},
                comptime anyview.as_view().reduce(dim).as_any_view(),
                dtype,
            );
        }

        pub fn sum(self: Self, comptime dim: u32) !Tensor(
            Self.dtype,
            anyview.as_view().reduce(dim).shape.*,
        ) {
            const node = sum_node(
                self.compiler.node_manager.get(self.index).?,
                dim,
            );
            const index = try self.compiler.node_manager.add(node);

            const reduce_shape = comptime anyview.as_view().reduce(dim).shape.*;
            return Tensor(Self.dtype, reduce_shape){
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

fn array_init_shape(comptime tuple: anytype) []const u32 {
    comptime {
        switch (@typeInfo(@TypeOf(tuple))) {
            .Struct, .Union, .Enum => {
                const fields = std.meta.fields(@TypeOf(tuple));

                var shape: [fields.len]u32 = undefined;

                for (fields, 0..) |field, i| {
                    switch (@typeInfo(field.type)) {
                        .Int, .ComptimeInt => {
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
            .Pointer => |info| {
                if (info.size == .Slice) {
                    switch (@typeInfo(info.child)) {
                        .Int, .ComptimeInt => {
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
                } else if (info.size == .One) {
                    return array_init_shape(tuple.*);
                } else {
                    @compileError(std.fmt.comptimePrint("Shape dims must be a tuple, slice, or array." ++
                        "  You passed in type {}", .{@TypeOf(tuple)}));
                }
            },
            .Array => |info| {
                switch (@typeInfo(info.child)) {
                    .Int, .ComptimeInt => {
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

// TODO: API is a little weird, accessing the function becomes A.mul(B) == Opertaions.mul(A, B)
pub fn Operations(comptime _dtype: dtypes.DType, comptime _anyview: view.AnyView, comptime _node: *const ast.Node) type {
    return extern struct {
        const Self = @This();

        const dtype = _dtype;
        const anyview = _anyview;
        pub const node = _node;

        scheduler: *Scheduler,
        node: *const ast.Node = node,

        pub fn init(scheduler: *Scheduler) Self {
            return Self{
                .scheduler = scheduler,
            };
        }

        // TODO: some of this schould be cached somehow,
        // schedules are cached thanks to the dep tree and such
        // ir gen should be too, codegen phase can probably as well
        pub fn realize(self: Self) !*RuntimeBuffer {
            const allocator = self.scheduler.allocator;

            try self.scheduler.mark_for_scheduling(node);
            const schedule = try self.scheduler.create_schedule(node);
            std.debug.print("{}\n", .{schedule});

            var buffer_map = std.AutoHashMap(usize, *RuntimeBuffer).init(allocator);
            defer buffer_map.deinit();

            for (schedule.global_buffers) |buffer| {
                try buffer_map.put(buffer.idx, buffer.buffer);
            }

            const ir_block = try IRGenerator.run(allocator, schedule);
            std.debug.print("{}\n", .{ir_block});

            // TODO: codegen? does this make sense for zig cpu runtime?
            try CPU.run(allocator, ir_block, &buffer_map);

            return buffer_map.get(0).?;
        }

        fn mul_node(comptime lhs: *const ast.Node, comptime rhs: *const ast.Node) *const ast.Node {
            comptime {
                return &ast.Node.init(
                    .Mul,
                    {},
                    .{ lhs, rhs },
                    &anyview,
                    dtype,
                );
            }
        }

        pub fn mul(self: Self, other: anytype) !Operations(
            dtype,
            anyview,
            mul_node(@TypeOf(self).node, to_meta_data(@TypeOf(other)).node),
        ) {
            binary_verify_shapes(anyview, to_meta_data(@TypeOf(other)).anyview);

            const self_node = node;
            const other_node = comptime to_meta_data(@TypeOf(other)).node;
            const n = comptime mul_node(self_node, other_node);
            return Operations(dtype, anyview, n).init(self.scheduler);
        }

        fn sum_node(
            comptime self: *const ast.Node,
            comptime dim: u32,
        ) *const ast.Node {
            comptime {
                return &ast.Node.init(
                    .Sum,
                    .{
                        .dim = dim,
                    },
                    .{self},
                    anyview.as_view().reduce(dim).as_any_view(),
                    dtype,
                );
            }
        }

        pub fn sum(self: Self, comptime dim: u32) !Operations(
            dtype,
            anyview.as_view().reduce(dim).as_any_view().*,
            sum_node(@TypeOf(self).node, dim),
        ) {
            const n = comptime sum_node(node, dim);
            return Operations(
                dtype,
                anyview.as_view().reduce(dim).as_any_view().*,
                n,
            ).init(self.scheduler);
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            _ = fmt;

            const buffer = try self.realize();

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
            for (0..self.node.view.size) |i| {
                const value_buffer = buffer.get(@intCast(i)).?;
                const value = std.mem.bytesToValue(comptime node.dtype.ToBuiltin(), value_buffer);

                try writer.print("{}, ", .{value});
            }
            try writer.writeAll("]\n");

            try writer.print(")", .{});
        }
    };
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
