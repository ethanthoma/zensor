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
        return View(self.shape);
    }

    pub fn to(comptime self: *const Self) Self.AsView(self.*) {
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
                const size = comptime blk: {
                    var size = 1;
                    for (shape) |dim| {
                        size *= dim;
                    }
                    break :blk size;
                };

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

            pub fn to_any_view(comptime self: *const Self) *const AnyView {
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
