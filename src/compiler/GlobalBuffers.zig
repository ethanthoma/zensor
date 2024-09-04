const std = @import("std");

const ast = @import("../ast.zig");
const dtypes = @import("../dtypes.zig");
const RuntimeBuffer = @import("../RuntimeBuffer.zig");

const Error = error{
    BufferNotAdded,
};

const GlobalBuffers = @This();

allocator: std.mem.Allocator,
buffers: std.AutoHashMap(*const ast.Node, *RuntimeBuffer),

pub fn init(allocator: std.mem.Allocator) GlobalBuffers {
    return .{
        .allocator = allocator,
        .buffers = std.AutoHashMap(*const ast.Node, *RuntimeBuffer).init(allocator),
    };
}

pub fn deinit(self: *GlobalBuffers) void {
    self.buffers.deinit();
}

pub fn register_buffer(self: *GlobalBuffers, node: *const ast.Node, buffer: *RuntimeBuffer) !void {
    try self.buffers.put(node, buffer);
}

pub fn get_buffer(self: *GlobalBuffers, node: *const ast.Node) !*RuntimeBuffer {
    return self.buffers.get(node) orelse Error.BufferNotAdded;
}

pub fn create_buffer(self: *GlobalBuffers, dtype: dtypes.DataType, shape: []const u32) !*RuntimeBuffer {
    const buffer = try self.allocator.create(RuntimeBuffer);
    errdefer self.allocator.destroy(buffer);

    buffer.* = try RuntimeBuffer.init(
        self.allocator,
        dtype,
        shape,
    );

    return buffer;
}
