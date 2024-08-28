const std = @import("std");
const FieldType = std.meta.FieldType;

pub const ast = @import("./ast.zig");

const v = @import("./view.zig");
pub const AnyView = v.AnyView;
pub const View = v.View;

pub const IRGenerator = @import("./IRGenerator.zig");

pub const dtypes = @import("./dtypes.zig");

pub const Scheduler = @import("./Scheduler.zig");

pub const buffer = @import("./buffer.zig");

// TODO: remove this and find a better way
pub const MemoryManager = struct {
    const Self = @This();

    buffers: std.AutoHashMapUnmanaged(ast.BufferID, buffer.RuntimeBuffer),

    pub fn init() Self {
        return Self{
            .buffers = .{},
        };
    }

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        self.buffers.deinit(allocator);
    }

    pub fn create_id(comptime T: type) ast.BufferID {
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

            var value: ast.BufferID = 0;
            for (type_name[index + 1 ..]) |c| {
                value = 10 * value + (@as(ast.BufferID, c) - '0');
            }
            return value;
        }
    }

    pub fn set_buffer(self: *Self, allocator: std.mem.Allocator, id: ast.BufferID, buf: buffer.RuntimeBuffer) !void {
        try self.buffers.putNoClobber(allocator, id, buf);
    }
};
