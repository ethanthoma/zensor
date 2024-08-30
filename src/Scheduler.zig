const std = @import("std");

const ast = @import("./ast.zig");
const RuntimeBuffer = @import("./RuntimeBuffer.zig");

const Scheduler = @This();

allocator: std.mem.Allocator,
scheduled_nodes: std.AutoHashMap(*const ast.Node, bool),
visited: std.AutoHashMap(*const ast.Node, void),
schedules: std.AutoHashMap(*const ast.Node, *const Schedule),
buffers: std.AutoHashMap(*const ast.Node, *RuntimeBuffer),

pub fn init(allocator: std.mem.Allocator) Scheduler {
    return .{
        .allocator = allocator,
        .scheduled_nodes = std.AutoHashMap(comptime *const ast.Node, bool).init(allocator),
        .visited = std.AutoHashMap(comptime *const ast.Node, void).init(allocator),
        .schedules = std.AutoHashMap(comptime *const ast.Node, *const Schedule).init(allocator),
        .buffers = std.AutoHashMap(comptime *const ast.Node, *RuntimeBuffer).init(allocator),
    };
}

pub fn schedule(self: *Scheduler, comptime ast_node: *const ast.Node) !void {
    try self.scheduled_nodes.put(comptime ast_node, false);
}

pub fn add_buffer(
    self: *Scheduler,
    comptime ast_node: *const ast.Node,
    buffer: *RuntimeBuffer,
) !void {
    try self.buffers.put(comptime ast_node, buffer);
}

pub fn run(self: *Scheduler, comptime ast_node: *const ast.Node) !Schedule {
    var order = std.ArrayList(*const ast.Node).init(self.allocator);
    var buffers = std.AutoHashMap(*const ast.Node, *RuntimeBuffer).init(self.allocator);
    var dependencies = std.ArrayList(*const Schedule).init(self.allocator);

    const final_node = comptime blk: {
        const node = ast.Node.init(
            .Store,
            {},
            [_]*const ast.Node{ast_node},
            ast_node.view,
            ast_node.dtype,
        );
        break :blk &node;
    };

    try self.finalize_schedule(final_node, &buffers);

    _ = try self.topological_sort(final_node, &order, &buffers, &dependencies, .{ .head = ast_node });

    const scheduled_node = Schedule.init(order.items, buffers, dependencies.items);

    try self.scheduled_nodes.put(ast_node, true);
    try self.schedules.put(ast_node, &scheduled_node);

    return scheduled_node;
}

fn finalize_schedule(
    self: *Scheduler,
    comptime store_node: *const ast.Node,
    buffers: *std.AutoHashMap(*const ast.Node, *RuntimeBuffer),
) !void {
    var buffer = try RuntimeBuffer.init(
        self.allocator,
        store_node.dtype,
        store_node.view.shape[0..store_node.view.rank],
    );
    errdefer buffer.deinit();
    try self.buffers.put(store_node, &buffer);
    try buffers.put(store_node, &buffer);
}

fn topological_sort(
    self: *Scheduler,
    comptime ast_node: *const ast.Node,
    order: *std.ArrayList(*const ast.Node),
    buffers: *std.AutoHashMap(*const ast.Node, *RuntimeBuffer),
    dependencies: *std.ArrayList(*const Schedule),
    context: struct { head: *const ast.Node },
) !*const ast.Node {
    if (false and self.is_dependency(context.head, ast_node)) {
        const dependency = try self.get_dependency(ast_node);
        try dependencies.append(dependency);

        const runtime_buffer_ptr = self.buffers.get(ast_node);

        const load_node = dependency_node(dependency);

        try self.buffers.put(load_node, runtime_buffer_ptr);
        try buffers.put(load_node, runtime_buffer_ptr);

        try self.visited.put(ast_node, {});
        try self.visited.put(load_node, {});

        return load_node;
    }

    if (self.visited.contains(ast_node)) {
        return ast_node;
    }

    try self.visited.put(ast_node, {});

    if (is_buffer_op(ast_node.op)) {
        const buffer = self.buffers.get(ast_node) orelse return error.BufferNotAdded;
        try buffers.put(ast_node, buffer);
    }

    const inputs = @field(ast_node.input, @tagName(ast_node.op));

    if (@typeInfo(@TypeOf(inputs)) == .Array) {
        inline for (inputs) |input| {
            ast_node.input = try self.topological_sort(input, order, buffers, dependencies, context);
        }
    }

    try order.append(ast_node);
    return ast_node;
}

fn dependency_node(dependency: *Schedule) *const ast.Node {
    const store_node = dependency.nodes[dependency.node.len - 1];

    const load_node: *const ast.Node = &ast.Node.init(
        .Load,
        {},
        [_]*const ast.Node{store_node},
        store_node.view,
        store_node.dtype,
    );

    return load_node;
}

fn is_dependency(self: *Scheduler, head: *const ast.Node, ast_node: *const ast.Node) bool {
    return self.scheduled_nodes.contains(ast_node) and head != ast_node;
}

fn get_dependency(self: *Scheduler, comptime ast_node: *const ast.Node) !*Schedule {
    const has_been_scheduled = self.scheduled_nodes.get(ast_node).?;

    if (!has_been_scheduled) {
        try self.scheduled_nodes.put(ast_node, true);
        return self.run(ast_node);
    }

    return self.schedules.get(ast_node).?;
}

fn is_buffer_op(op: ast.Operations) bool {
    return switch (op) {
        .Load, .Store => true,
        else => false,
    };
}

fn ast_node_eql(lhs: *const ast.Node, rhs: *const ast.Node) bool {
    return lhs == rhs;
}

fn buffer_id_eql(lhs: BufferContext, rhs: BufferContext) bool {
    return lhs.id == rhs.id;
}

const BufferContext = struct {
    id: ast.BufferID,
    writable: bool,
};

pub const Schedule = struct {
    nodes: []const *const ast.Node,
    global_buffers: std.AutoHashMap(*const ast.Node, *RuntimeBuffer),
    dependencies: []const *const Schedule,
    ran: bool = false,

    pub fn init(
        nodes: []const *const ast.Node,
        global_buffers: std.AutoHashMap(*const ast.Node, *RuntimeBuffer),
        dependencies: []const *const Schedule,
    ) @This() {
        return .{
            .nodes = nodes,
            .global_buffers = global_buffers,
            .dependencies = dependencies,
        };
    }

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;
        try writer.print("{{[{}]ast.Nodes, ", .{self.nodes.len});
        try writer.print("[", .{});
        var iter = self.global_buffers.iterator();
        var first = true;
        while (iter.next()) |entry| {
            const op = entry.key_ptr.*.op;
            const runtime_buffer_ptr = entry.value_ptr.*;

            if (!first) {
                try writer.print(", ", .{});
            }

            try writer.print("{}@{x}", .{ op, @intFromPtr(runtime_buffer_ptr) });

            first = false;
        }
        try writer.print("]}}", .{});
    }
};
