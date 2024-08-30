const std = @import("std");

pub const AnyView = extern struct {
    const Self = @This();

    shape: [*]u32,
    strides: [*]u32,
    offset: u32,
    mask: ?[*]u32,
    contiguous: bool,
    rank: u32,
    size: u32,

    fn AsView(comptime self: *const Self) type {
        return View(self.shape[0..self.rank]);
    }

    pub fn as_view(comptime self: *const Self) *const Self.AsView(self) {
        return @ptrCast(self);
    }
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

        const size = blk: {
            var size = 1;
            for (shape) |dim| {
                size *= dim;
            }
            break :blk size;
        };

        return extern struct {
            const Self = @This();

            shape: *const [rank]u32,
            strides: *const [rank]u32,
            offset: u32,
            mask: ?*const [rank]u32,
            contiguous: bool,
            rank: u32,
            size: u32,

            pub fn init() Self {
                return Self{
                    .shape = shape[0..],
                    .strides = strides[0..],
                    .offset = 0,
                    .mask = null,
                    .contiguous = true,
                    .rank = rank,
                    .size = size,
                };
            }

            pub fn as_any_view(comptime self: *const Self) *const AnyView {
                comptime {
                    return @ptrCast(self);
                }
            }

            pub fn Reduce(comptime self: *const Self, comptime dim: u32) type {
                if (dim >= rank) @compileError("Dimension out of bounds");

                const new_shape: []const u32 = comptime blk: {
                    var new_shape: [rank]u32 = self.shape.*;
                    new_shape[dim] = 1;
                    const final_shape = new_shape;
                    break :blk final_shape[0..];
                };

                return View(new_shape);
            }

            pub fn reduce(comptime self: *const Self, comptime dim: u32) self.Reduce(dim) {
                comptime {
                    if (dim >= rank) @compileError("Dimension out of bounds");

                    var new_shape: [rank]u32 = self.shape.*;
                    new_shape[dim] = 1;
                    const final_shape = new_shape;

                    var new_strides: [rank]u32 = self.strides.*;
                    new_strides[dim] = 0; // Set stride to 0 for the reduced dimension
                    const final_strides = new_strides;

                    const new_size: u32 = self.size / self.shape[dim];

                    return self.Reduce(dim){
                        .shape = final_shape[0..],
                        .strides = final_strides[0..],
                        .offset = self.offset,
                        .mask = self.mask,
                        .contiguous = false, // Reducing a dimension typically makes the view non-contiguous
                        .rank = rank,
                        .size = new_size,
                    };
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
