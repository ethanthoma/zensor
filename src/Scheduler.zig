const std = @import("std");

const ast = @import("./ast.zig");
const dtypes = @import("./dtypes.zig");
const view = @import("./view.zig");
const RuntimeBuffer = @import("./RuntimeBuffer.zig");
const Schedule = @import("./Schedule.zig");

const Scheduler = @This();

const Error = error{
    BufferNotAdded,
    NodeNotScheduled,
};

pub const NodeStatus = enum {
    Unscheduled,
    Scheduled,
    Processed,
};

allocator: std.mem.Allocator,
node_statuses: std.AutoHashMap(*const ast.Node, NodeStatus),
schedules: std.AutoHashMap(*const ast.Node, *const Schedule),
buffers: std.AutoHashMap(*const ast.Node, *RuntimeBuffer),

pub fn init(allocator: std.mem.Allocator) Scheduler {
    return .{
        .allocator = allocator,
        .node_statuses = std.AutoHashMap(*const ast.Node, NodeStatus).init(allocator),
        .schedules = std.AutoHashMap(*const ast.Node, *const Schedule).init(allocator),
        .buffers = std.AutoHashMap(*const ast.Node, *RuntimeBuffer).init(allocator),
    };
}

pub fn deinit(self: *Scheduler) void {
    self.node_statuses.deinit();
    self.schedules.deinit();
    self.buffers.deinit();
}

pub fn mark_for_scheduling(self: *Scheduler, comptime node: *const ast.Node) !void {
    try self.node_statuses.put(node, .Unscheduled);
}

pub fn add_buffer(
    self: *Scheduler,
    comptime node: *const ast.Node,
    buffer: *RuntimeBuffer,
) !void {
    try self.buffers.put(node, buffer);
}

const Context = struct {
    head: *const ast.Node,
    buffers: std.AutoArrayHashMap(*const ast.Node, Schedule.BufferContext),
    dependencies: std.ArrayList(*const Schedule),
    order: std.ArrayList(*const ast.Node),
    visited: std.AutoHashMap(*const ast.Node, *const ast.Node),

    pub fn init(allocator: std.mem.Allocator, comptime head: *const ast.Node) !@This() {
        return .{
            .head = head,
            .buffers = std.AutoArrayHashMap(*const ast.Node, Schedule.BufferContext).init(allocator),
            .dependencies = std.ArrayList(*const Schedule).init(allocator),
            .order = std.ArrayList(*const ast.Node).init(allocator),
            .visited = std.AutoHashMap(*const ast.Node, *const ast.Node).init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.buffers.deinit();
        self.dependencies.deinit();
        self.order.deinit();
        self.visited.deinit();
    }
};

pub fn run(
    self: *Scheduler,
    comptime root: *const ast.Node,
) (Error || std.mem.Allocator.Error)!*const Schedule {
    if (self.schedules.get(root)) |scheduled_node| {
        return scheduled_node;
    }

    var context = try Context.init(self.allocator, root);
    defer context.deinit();

    const last_node = try self.topological_sort(root, &context);

    try self.finalize_schedule(last_node, &context);

    const scheduled_node = try Schedule.from_slices(
        self.allocator,
        context.order.items,
        context.buffers.values(),
        context.dependencies.items,
    );
    errdefer scheduled_node.deinit();

    try self.node_statuses.put(root, .Processed);
    try self.schedules.put(root, scheduled_node);

    return scheduled_node;
}

fn topological_sort(
    self: *Scheduler,
    comptime ast_node: *const ast.Node,
    context: *Context,
) (Error || std.mem.Allocator.Error)!*const ast.Node {
    if (self.is_dependency(ast_node, context)) {
        return self.handle_dependency(ast_node, context);
    }

    if (context.visited.get(ast_node)) |node| {
        return node;
    }

    const cur_node = try self.create_runtime_node(ast_node, context);

    if (cur_node.op.AsOperationType() == .Buffer) {
        const buffer = self.buffers.get(ast_node) orelse return Error.BufferNotAdded;
        const name = switch (cur_node.op) {
            .Load => cur_node.arg.Load.name,
            .Store => cur_node.arg.Load.name,
            else => unreachable,
        };
        try context.buffers.put(cur_node, .{ .name = name, .buffer = buffer });
    }

    if (@typeInfo(std.meta.FieldType(ast.Operation.Input, ast_node.op)) == .Array) {
        const comptime_inputs = @field(ast_node.input, @tagName(ast_node.op));

        inline for (comptime_inputs, 0..) |input, i| {
            const node = try self.topological_sort(input, context);
            @field(cur_node.input, @tagName(ast_node.op))[i] = node;
        }
    }

    try context.order.append(cur_node);
    return cur_node;
}

fn is_dependency(self: *Scheduler, node: *const ast.Node, context: *Context) bool {
    return self.node_statuses.contains(node) and context.head != node;
}

fn handle_dependency(self: *Scheduler, comptime ast_node: *const ast.Node, context: *Context) !*const ast.Node {
    const dependency = try self.get_or_create_schedule(ast_node);
    try context.dependencies.append(dependency);

    const store_node = dependency.nodes[dependency.nodes.len - 1];
    const load_node = try create_load_node(self, store_node, context);

    try context.visited.put(ast_node, load_node);

    try context.order.append(load_node);
    return load_node;
}

fn get_or_create_schedule(self: *Scheduler, comptime node: *const ast.Node) !*const Schedule {
    if (self.schedules.get(node)) |_schedule| {
        return _schedule;
    }

    return try self.run(node);
}

fn create_load_node(self: *Scheduler, store_node: *const ast.Node, context: *Context) !*ast.Node {
    const load_node = try self.allocator.create(ast.Node);
    errdefer self.allocator.destroy(load_node);
    load_node.* = ast.Node.init(
        .Load,
        .{ .name = store_node.arg.Store.name },
        {},
        store_node.view,
        store_node.dtype,
    );

    const buffer = self.buffers.get(store_node) orelse return Error.BufferNotAdded;
    try self.buffers.put(load_node, buffer);
    try context.buffers.put(load_node, .{ .name = store_node.arg.Store.name, .buffer = buffer });

    return load_node;
}

fn create_runtime_node(self: *Scheduler, ast_node: *const ast.Node, context: *Context) !*ast.Node {
    const cur_node = try self.allocator.create(ast.Node);
    cur_node.* = ast_node.*;
    try context.visited.put(ast_node, cur_node);
    return cur_node;
}

fn finalize_schedule(
    self: *Scheduler,
    last_node: *const ast.Node,
    context: *Context,
) !void {
    const node = try self.allocator.create(ast.Node);
    node.* = ast.Node.init(
        .Store,
        .{ .name = try std.fmt.allocPrint(self.allocator, "{}", .{@intFromPtr(last_node)}) },
        [_]*const ast.Node{last_node},
        last_node.view,
        last_node.dtype,
    );
    errdefer self.allocator.destroy(node);
    try context.order.append(node);

    const buffer = try self.create_buffer(node);
    errdefer buffer.deinit();
    try self.buffers.put(node, buffer);
    try context.buffers.put(node, .{ .name = node.arg.Store.name, .buffer = buffer });
}

fn create_buffer(self: *Scheduler, node: *const ast.Node) !*RuntimeBuffer {
    const buffer = try self.allocator.create(RuntimeBuffer);
    errdefer self.allocator.destroy(buffer);

    buffer.* = try RuntimeBuffer.init(
        self.allocator,
        node.dtype,
        node.view.shape[0..node.view.rank],
    );

    return buffer;
}
