const std = @import("std");

const ast = @import("../ast.zig");
const dtypes = @import("../dtypes.zig");
const view = @import("../view.zig");
const RuntimeBuffer = @import("../RuntimeBuffer.zig");

const Schedule = @import("./Schedule.zig");
const GlobalBuffers = @import("./GlobalBuffers.zig");

const Scheduler = @This();

const Error = error{
    BufferNotAdded,
};

pub const NodeStatus = enum {
    Unscheduled,
    Scheduled,
    Processed,
};

const Context = struct {
    head: *const ast.Node,
    buffers: std.ArrayList(Schedule.BufferContext),
    dependencies: std.ArrayList(*const Schedule),
    order: std.ArrayList(*const ast.Node),
    visited: std.AutoHashMap(*const ast.Node, *const ast.Node),

    pub fn init(allocator: std.mem.Allocator, comptime head: *const ast.Node) !@This() {
        return .{
            .head = head,
            .buffers = std.ArrayList(Schedule.BufferContext).init(allocator),
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

allocator: std.mem.Allocator,
node_statuses: std.AutoHashMap(*const ast.Node, NodeStatus),
schedules: std.AutoHashMap(*const ast.Node, *const Schedule),
global_buffers: GlobalBuffers,

pub fn init(allocator: std.mem.Allocator) Scheduler {
    return .{
        .allocator = allocator,
        .node_statuses = std.AutoHashMap(*const ast.Node, NodeStatus).init(allocator),
        .schedules = std.AutoHashMap(*const ast.Node, *const Schedule).init(allocator),
        .global_buffers = GlobalBuffers.init(allocator),
    };
}

pub fn deinit(self: *Scheduler) void {
    self.node_statuses.deinit();
    self.schedules.deinit();
    self.global_buffers.deinit();
}

pub fn mark_for_scheduling(self: *Scheduler, comptime node: *const ast.Node) !void {
    try self.node_statuses.put(node, .Unscheduled);
}

pub fn register_buffer(
    self: *Scheduler,
    comptime node: *const ast.Node,
    buffer: *RuntimeBuffer,
) !void {
    try self.global_buffers.register_buffer(node, buffer);
}

pub fn run(
    self: *Scheduler,
    comptime root: *const ast.Node,
) (Error || std.mem.Allocator.Error)!*const Schedule {
    if (self.schedules.get(root)) |scheduled_node| {
        return scheduled_node;
    }

    var context = try Context.init(self.allocator, root);
    defer context.deinit();

    const buffer = try self.global_buffers.create_buffer(
        root.dtype,
        root.view.shape[0..root.view.rank],
    );
    errdefer buffer.deinit();

    const name = try std.fmt.allocPrint(self.allocator, "{x}", .{@intFromPtr(root)});
    errdefer self.allocator.free(name);

    try context.buffers.append(.{
        .name = name,
        .idx = context.buffers.items.len,
        .buffer = buffer,
        .writable = true,
    });

    const last_node = try self.topological_sort(root, &context);
    const node = try self.create_store_node(last_node, name, buffer, &context);
    node.input.Store[0] = last_node;

    const scheduled_node = try Schedule.from_slices(
        self.allocator,
        context.order.items,
        context.buffers.items,
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
        try self.handle_buffer_node(ast_node, cur_node, context);
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

fn handle_buffer_node(
    self: *Scheduler,
    comptime node: *const ast.Node,
    cur_node: *const ast.Node,
    context: *Context,
) !void {
    const buffer = try self.global_buffers.get_buffer(node);

    const name = switch (cur_node.op) {
        .Load => cur_node.arg.Load.name,
        .Store => cur_node.arg.Load.name,
        else => unreachable,
    };

    try context.buffers.append(.{
        .name = name,
        .idx = context.buffers.items.len,
        .buffer = buffer,
        .writable = false,
    });
}

fn is_dependency(self: *Scheduler, node: *const ast.Node, context: *Context) bool {
    return self.node_statuses.contains(node) and context.head != node;
}

fn handle_dependency(self: *Scheduler, comptime node: *const ast.Node, context: *Context) !*const ast.Node {
    const dependency = self.schedules.get(node) orelse try self.run(node);
    try context.dependencies.append(dependency);

    const store_node = dependency.nodes[dependency.nodes.len - 1];
    const load_node = try create_load_node(self, store_node, context);

    try context.visited.put(node, load_node);

    try context.order.append(load_node);
    return load_node;
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

    const buffer = try self.global_buffers.get_buffer(store_node);
    try self.global_buffers.register_buffer(load_node, buffer);

    try context.buffers.append(.{
        .name = store_node.arg.Store.name,
        .idx = context.buffers.items.len,
        .buffer = buffer,
        .writable = false,
    });

    return load_node;
}

fn create_runtime_node(self: *Scheduler, ast_node: *const ast.Node, context: *Context) !*ast.Node {
    const cur_node = try self.allocator.create(ast.Node);
    cur_node.* = ast_node.*;
    try context.visited.put(ast_node, cur_node);
    return cur_node;
}

fn create_store_node(
    self: *Scheduler,
    load_node: *const ast.Node,
    name: []const u8,
    buffer: *RuntimeBuffer,
    context: *Context,
) !*ast.Node {
    const node = try self.allocator.create(ast.Node);
    errdefer self.allocator.destroy(node);

    node.* = ast.Node.init(
        .Store,
        .{ .name = name },
        [_]*const ast.Node{load_node},
        load_node.view,
        load_node.dtype,
    );

    try self.global_buffers.register_buffer(node, buffer);

    try context.order.append(node);
    return node;
}
