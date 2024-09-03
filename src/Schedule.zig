const std = @import("std");

const ast = @import("./ast.zig");
const RuntimeBuffer = @import("./RuntimeBuffer.zig");

const Schedule = @This();

allocator: std.mem.Allocator,
nodes: []const *const ast.Node,
global_buffers: []const BufferContext,
dependencies: []const *const Schedule,
schedule_status: ScheduleStatus = .NotRun,

pub const ScheduleStatus = enum {
    NotRun,
    Running,
    Completed,
};

pub fn from_slices(
    allocator: std.mem.Allocator,
    nodes: []const *const ast.Node,
    global_buffers: []const BufferContext,
    dependencies: []const *const Schedule,
) !*const Schedule {
    const self = try allocator.create(@This());
    errdefer allocator.destroy(self);

    self.* = .{
        .allocator = allocator,
        .nodes = try allocator.dupe(*const ast.Node, nodes),
        .global_buffers = try allocator.dupe(BufferContext, global_buffers),
        .dependencies = try allocator.dupe(*const Schedule, dependencies),
    };

    return self;
}

pub fn deinit(self: *const Schedule) void {
    self.allocator.free(self.nodes);
    self.allocator.free(self.global_buffers);
    self.allocator.free(self.dependencies);
    self.allocator.destroy(self);
}

pub fn format(
    self: Schedule,
    comptime fmt: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    _ = options;
    _ = fmt;
    try writer.writeAll("Schedule{\n");

    try writer.print("\tstatus: {s}\n", .{@tagName(self.schedule_status)});

    try writer.print("\ttopological sort: [{}]ast.Nodes{{", .{self.nodes.len});
    for (self.nodes, 0..) |node, i| {
        try writer.print("{s}", .{@tagName(node.op)});

        if (i < self.nodes.len - 1) {
            try writer.writeAll(", ");
        }
    }
    try writer.writeAll("}, \n");

    try writer.writeAll("\tglobal buffer names: [");
    for (self.global_buffers, 0..) |buffer_context, i| {
        try writer.writeAll("\"");
        try writer.writeAll(buffer_context.name);
        try writer.writeAll("\"");

        if (i < self.global_buffers.len - 1) {
            try writer.writeAll(", ");
        }
    }
    try writer.writeAll("],\n");

    try writer.print("\tdependencies count: {},\n", .{self.dependencies.len});

    try writer.print("\tAST:\n{t1},\n", .{self.nodes[self.nodes.len - 1]});

    try writer.writeAll("}");
}

pub const BufferContext = struct {
    name: []const u8,
    buffer: *RuntimeBuffer,
};
