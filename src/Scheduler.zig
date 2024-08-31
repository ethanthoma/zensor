const std = @import("std");

const ast = @import("./ast.zig");
const dtypes = @import("./dtypes.zig");
const view = @import("./view.zig");
const RuntimeBuffer = @import("./RuntimeBuffer.zig");

const Scheduler = @This();

allocator: std.mem.Allocator,
scheduled_nodes: std.AutoHashMap(*const ast.Node, bool),
visited: std.AutoHashMap(*const ast.Node, usize),
schedules: std.AutoHashMap(*const ast.Node, *const Schedule),
buffers: std.AutoHashMap(*const ast.Node, *RuntimeBuffer),
runtime_nodes: std.ArrayList(*const ast.Node),

pub fn init(allocator: std.mem.Allocator) Scheduler {
    return .{
        .allocator = allocator,
        .scheduled_nodes = std.AutoHashMap(*const ast.Node, bool).init(allocator),
        .visited = std.AutoHashMap(*const ast.Node, usize).init(allocator),
        .schedules = std.AutoHashMap(*const ast.Node, *const Schedule).init(allocator),
        .buffers = std.AutoHashMap(*const ast.Node, *RuntimeBuffer).init(allocator),
        .runtime_nodes = std.ArrayList(*const ast.Node).init(allocator),
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
    try self.buffers.put(ast_node, buffer);
}

pub fn run(
    self: *Scheduler,
    comptime ast_node: *const ast.Node,
) (Error || std.mem.Allocator.Error)!*const Schedule {
    if (self.schedules.get(ast_node)) |scheduled_node| {
        return scheduled_node;
    }

    var order = std.ArrayList(*const ast.Node).init(self.allocator);
    var buffers = std.AutoHashMap(*const ast.Node, *RuntimeBuffer).init(self.allocator);
    var dependencies = std.ArrayList(*const Schedule).init(self.allocator);

    const last_node = try self.topological_sort(ast_node, &order, &buffers, &dependencies, .{ .head = ast_node });

    try self.finalize_schedule(
        last_node,
        &order,
        &buffers,
    );

    const scheduled_node = try self.allocator.create(Schedule);
    scheduled_node.* = Schedule.init(order.items, buffers, dependencies.items);

    try self.scheduled_nodes.put(ast_node, true);
    try self.schedules.put(ast_node, scheduled_node);

    return scheduled_node;
}

fn finalize_schedule(
    self: *Scheduler,
    ast_node: *const ast.Node,
    order: *std.ArrayList(*const ast.Node),
    buffers: *std.AutoHashMap(*const ast.Node, *RuntimeBuffer),
) !void {
    const node = try self.allocator.create(ast.Node);
    node.* = ast.Node.init(
        .Store,
        .{ .name = try std.fmt.allocPrint(self.allocator, "{}", .{@intFromPtr(ast_node)}) },
        [_]*const ast.Node{ast_node},
        ast_node.view,
        ast_node.dtype,
    );
    errdefer self.allocator.destroy(node);
    try self.runtime_nodes.append(node);
    try order.append(node);

    const buffer = try self.allocator.create(RuntimeBuffer);
    buffer.* = try RuntimeBuffer.init(
        self.allocator,
        node.dtype,
        node.view.shape[0..node.view.rank],
    );
    errdefer buffer.deinit();
    try self.buffers.put(node, buffer);
    try buffers.put(node, buffer);
}

const Error = error{
    BufferNotAdded,
};

fn topological_sort(
    self: *Scheduler,
    comptime ast_node: *const ast.Node,
    order: *std.ArrayList(*const ast.Node),
    buffers: *std.AutoHashMap(*const ast.Node, *RuntimeBuffer),
    dependencies: *std.ArrayList(*const Schedule),
    context: struct { head: *const ast.Node },
) (Error || std.mem.Allocator.Error)!*const ast.Node {
    if (self.is_dependency(context.head, ast_node)) {
        const dependency: *const Schedule = try self.get_dependency(ast_node);
        try dependencies.append(dependency);

        const store_node: *const ast.Node = dependency.nodes[dependency.nodes.len - 1];
        const runtime_buffer_ptr = self.buffers.get(store_node) orelse return Error.BufferNotAdded;

        const load_node = try self.allocator.create(ast.Node);
        load_node.* = ast.Node.init(
            .Load,
            .{ .name = store_node.arg.Store.name },
            {},
            store_node.view,
            store_node.dtype,
        );
        try self.runtime_nodes.append(load_node);

        try self.buffers.put(load_node, runtime_buffer_ptr);
        try buffers.put(load_node, runtime_buffer_ptr);

        try self.visited.put(ast_node, self.runtime_nodes.items.len - 1);

        try order.append(load_node);

        return load_node;
    }

    if (self.visited.contains(ast_node)) {
        const index = self.visited.get(ast_node).?;
        return self.runtime_nodes.items[index];
    }

    const cur_node = try self.allocator.create(ast.Node);
    cur_node.* = ast_node.*;
    try self.runtime_nodes.append(cur_node);

    try self.visited.put(ast_node, self.runtime_nodes.items.len - 1);

    if (is_buffer_op(cur_node.op)) {
        const buffer = self.buffers.get(ast_node) orelse return Error.BufferNotAdded;
        try buffers.put(cur_node, buffer);
    }

    // TODO: clean this up
    switch (ast_node.input) {
        .Sum => |inputs| {
            inline for (inputs, 0..) |input, i| {
                const node = try self.topological_sort(input, order, buffers, dependencies, context);
                cur_node.input.Sum[i] = node;
            }
        },
        .Mul => |inputs| {
            inline for (inputs, 0..) |input, i| {
                const node = try self.topological_sort(input, order, buffers, dependencies, context);
                cur_node.input.Mul[i] = node;
            }
        },
        .Store => |inputs| {
            inline for (inputs, 0..) |input, i| {
                const node = try self.topological_sort(input, order, buffers, dependencies, context);
                cur_node.input.Store[i] = node;
            }
        },

        else => {},
    }

    try order.append(cur_node);
    return cur_node;
}

fn dependency_node(dependency: *const Schedule) *const ast.Node {
    const store_node: *const ast.Node = dependency.nodes[dependency.nodes.len - 1];

    const load_node: *const ast.Node = &ast.Node.init(
        .Load,
        .{ .name = store_node.arg.Store.name },
        {},
        store_node.view,
        store_node.dtype,
    );

    return load_node;
}

fn is_dependency(self: *Scheduler, head: *const ast.Node, ast_node: *const ast.Node) bool {
    return self.scheduled_nodes.contains(ast_node) and head != ast_node;
}

fn get_dependency(self: *Scheduler, comptime ast_node: *const ast.Node) !*const Schedule {
    if (self.schedules.get(ast_node)) |_schedule| {
        return _schedule;
    }

    try self.scheduled_nodes.put(ast_node, true);
    return try self.run(ast_node);
}

fn is_buffer_op(op: ast.Operations) bool {
    return switch (op) {
        .Load, .Store => true,
        else => false,
    };
}

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
        try writer.writeAll("Schedule {\n");

        try writer.print("topological sort: [{}]ast.Nodes{{", .{self.nodes.len});
        for (self.nodes, 0..) |node, i| {
            try writer.print("{s}", .{@tagName(node.op)});

            if (i < self.nodes.len - 1) {
                try writer.writeAll(", ");
            }
        }
        try writer.writeAll("}, \n");

        try writer.writeAll("global buffers: [");
        var iter = self.global_buffers.iterator();
        var first = true;
        while (iter.next()) |entry| {
            const op = entry.key_ptr.*.op;
            const runtime_buffer_ptr = entry.value_ptr.*;

            if (!first) {
                try writer.print(", ", .{});
            }

            try writer.print("{s}@{x}", .{ @tagName(op), @intFromPtr(runtime_buffer_ptr) });

            first = false;
        }
        try writer.writeAll("],\n");

        try writer.print("dependencies count: {},\n", .{self.dependencies.len});

        try writer.print("AST:\n{},\n", .{self.nodes[self.nodes.len - 1]});

        try writer.writeAll("}");
    }
};
