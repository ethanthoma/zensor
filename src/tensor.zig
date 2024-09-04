const std = @import("std");

const ast = @import("./ast.zig");
const dtypes = @import("./dtypes.zig");
const view = @import("./view.zig");
const Scheduler = @import("./compiler/Scheduler.zig");
const RuntimeBuffer = @import("./RuntimeBuffer.zig");
const Device = @import("./backend.zig").Device;

/// Handles Tensor creation.
/// Maybe there is some abuse of usingnamespace to merge this and the Operations
/// struct below.
pub fn Tensor(comptime _dtype: dtypes.DataType, comptime _shape: []const u32) type {
    const anyview = view.View(_shape).init().as_any_view();

    return struct {
        fn full_node(comptime value: anytype) *const ast.Node {
            comptime {
                return &ast.Node.init(
                    .Const,
                    .{ .value = std.fmt.comptimePrint("{}", .{value}) },
                    {},
                    anyview,
                    _dtype,
                );
            }
        }

        pub fn full(scheduler: *Scheduler, comptime value: anytype) !Operations(
            _dtype,
            anyview.*,
            full_node(value),
        ) {
            return Operations(
                _dtype,
                anyview.*,
                full_node(value),
            ).init(scheduler);
        }

        fn load_node(comptime name: []const u8) *const ast.Node {
            comptime {
                return &ast.Node.init(
                    .Load,
                    .{ .name = name },
                    {},
                    anyview,
                    _dtype,
                );
            }
        }

        fn str_to_hex(comptime str: []const u8) []const u8 {
            comptime {
                var hex: []const u8 = "";
                for (str) |char| {
                    hex = hex ++ std.fmt.comptimePrint("{x}", .{char});
                }
                return hex;
            }
        }

        pub fn from_numpy(scheduler: *Scheduler, comptime filename: []const u8) !Operations(
            _dtype,
            anyview.*,
            load_node(str_to_hex(filename)),
        ) {
            const node = comptime load_node(str_to_hex(filename));
            const buffer = try scheduler.allocator.create(RuntimeBuffer);
            errdefer scheduler.allocator.destroy(buffer);
            buffer.* = try RuntimeBuffer.Numpy.load(scheduler.allocator, filename);
            try scheduler.register_buffer(node, buffer);
            return Operations(
                _dtype,
                anyview.*,
                node,
            ).init(scheduler);
        }
    };
}

const Metadata = struct {
    dtype: dtypes.DataType,
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
pub fn Operations(comptime _dtype: dtypes.DataType, comptime _anyview: view.AnyView, comptime _node: *const ast.Node) type {
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

        /// Adds the node to the scheduler. Should be known at comptime.
        /// The user should never need to call this.
        pub fn realize(self: Self, device: Device) !void {
            try self.scheduler.mark_for_scheduling(node);
            const schedules = try self.scheduler.run(node);

            std.debug.print("{s}\n{}\n", .{ @tagName(device), schedules });
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

            try self.realize();

            try writer.print("Tensor(\n", .{});

            try writer.print("\ttype: {},\n", .{dtype});

            try writer.print("\tshape: [", .{});
            for (anyview.shape, 0..anyview.rank) |dim, i| {
                if (i > 0) try writer.print(", ", .{});
                try writer.print("{}", .{dim});
            }
            try writer.print("],\n", .{});

            try writer.print("\tlength: {},\n", .{anyview.size});

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
            if (@TypeOf(Context.dtype) != dtypes.DataType) {
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
