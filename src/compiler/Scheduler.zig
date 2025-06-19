const std = @import("std");

const ast = @import("ast.zig");
const dtypes = @import("../dtypes.zig");
const view = @import("../view.zig");
const GlobalBuffers = @import("GlobalBuffers.zig");
const RuntimeBuffer = @import("../RuntimeBuffer.zig");
const Schedule = @import("Schedule.zig");

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
    head: *ast.Node,
    buffers: std.AutoArrayHashMap(*RuntimeBuffer, Schedule.BufferContext),
    dependencies: std.ArrayList(*Schedule),
    order: std.ArrayList(*ast.Node),
    visited: std.AutoHashMap(*ast.Node, void),

    pub fn init(allocator: std.mem.Allocator, head: *ast.Node) !@This() {
        return .{
            .head = head,
            .buffers = std.AutoArrayHashMap(*RuntimeBuffer, Schedule.BufferContext).init(allocator),
            .dependencies = std.ArrayList(*Schedule).init(allocator),
            .order = std.ArrayList(*ast.Node).init(allocator),
            .visited = std.AutoHashMap(*ast.Node, void).init(allocator),
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
node_statuses: std.AutoHashMap(*ast.Node, NodeStatus),
schedules: std.AutoArrayHashMap(*ast.Node, *Schedule),

pub fn init(allocator: std.mem.Allocator) Scheduler {
    return .{
        .allocator = allocator,
        .node_statuses = std.AutoHashMap(*ast.Node, NodeStatus).init(allocator),
        .schedules = std.AutoArrayHashMap(*ast.Node, *Schedule).init(allocator),
    };
}

pub fn deinit(self: *Scheduler) void {
    self.node_statuses.deinit();
    self.schedules.deinit();
}

pub fn mark_for_scheduling(self: *Scheduler, node: *ast.Node) !void {
    try self.node_statuses.put(node, .Unscheduled);
}

pub fn create_schedule(
    self: *Scheduler,
    root: *ast.Node,
) (Error || std.mem.Allocator.Error)!void {
    if (self.node_statuses.get(root)) |status| {
        switch (status) {
            .Scheduled => return,
            .Unscheduled => {},
            .Processed => {},
        }
    }

    var context = try Context.init(self.allocator, root);
    defer context.deinit();

    const final_node = try self.create_final_node(&context);
    _ = try self.topological_sort(final_node, &context);

    const schedule = try Schedule.from_slices(
        self.allocator,
        context.order.items,
        context.buffers.values(),
        context.dependencies.items,
    );
    errdefer schedule.deinit();

    try self.node_statuses.put(root, .Scheduled);
    try self.schedules.put(root, schedule);
}

fn create_final_node(self: *Scheduler, context: *Context) !*ast.Node {
    const buffer = try self.allocator.create(RuntimeBuffer);
    errdefer self.allocator.destroy(buffer);

    buffer.* = try RuntimeBuffer.init(
        self.allocator,
        context.head.dtype,
        context.head.view.shape[0..context.head.view.rank],
    );
    errdefer buffer.deinit();

    return try self.create_store_node(context.head, buffer);
}

fn topological_sort(
    self: *Scheduler,
    node: *ast.Node,
    context: *Context,
) (Error || std.mem.Allocator.Error)!*ast.Node {
    if (self.is_dependency(node, context)) {
        return try self.handle_dependency(node, context);
    }

    if (context.visited.contains(node)) {
        return node;
    }

    if (node.op.AsOperationType() == .Buffer) {
        const buffer = switch (node.arg) {
            inline .Load, .Store => |arg| arg.buffer,
            else => unreachable,
        };

        const writable = switch (node.op) {
            .Load => false,
            .Store => true,
            else => unreachable,
        };

        try context.buffers.put(buffer, .{
            .idx = context.buffers.values().len,
            .ptr = buffer,
            .writable = writable,
        });
    }

    switch (node.input) {
        inline else => |*inputs| {
            const info = @typeInfo(@TypeOf(inputs));
            if (info == .pointer and @typeInfo(info.pointer.child) == .array) {
                for (inputs) |*input| {
                    input.* = try self.topological_sort(input.*, context);
                }
            }
        },
    }

    try context.order.append(node);
    try context.visited.put(node, {});
    return node;
}

fn is_dependency(self: *Scheduler, node: *ast.Node, context: *Context) bool {
    return self.node_statuses.contains(node) and context.head != node;
}

fn handle_dependency(self: *Scheduler, node: *ast.Node, context: *Context) !*ast.Node {
    try self.create_schedule(node);
    const dependency = self.schedules.get(node).?;
    try context.dependencies.append(dependency);

    const store_node = dependency.nodes[dependency.nodes.len - 1];
    const load_node = try create_load_node(self, store_node, context);

    try context.visited.put(load_node, {});
    try context.visited.put(node, {});
    try context.order.append(load_node);

    return load_node;
}

fn create_load_node(self: *Scheduler, store_node: *const ast.Node, context: *Context) !*ast.Node {
    const buffer = store_node.arg.Store.buffer;

    const load_node = try self.allocator.create(ast.Node);
    errdefer self.allocator.destroy(load_node);
    load_node.* = ast.Node.init(
        .Load,
        .{ .buffer = buffer },
        {},
        store_node.view,
        store_node.dtype,
    );

    try context.buffers.put(buffer, .{
        .idx = context.buffers.values().len,
        .ptr = buffer,
        .writable = false,
    });

    return load_node;
}

fn create_store_node(
    self: *Scheduler,
    node: *ast.Node,
    buffer: *RuntimeBuffer,
) !*ast.Node {
    const store_node = try self.allocator.create(ast.Node);
    errdefer self.allocator.destroy(node);
    store_node.* = ast.Node.init(
        .Store,
        .{ .buffer = buffer },
        [_]*ast.Node{node},
        node.view,
        node.dtype,
    );

    return store_node;
}

pub fn fetch_schedules(self: Scheduler, node: *ast.Node) ![]*Schedule {
    var schedules = std.ArrayList(*Schedule).init(self.allocator);
    defer schedules.deinit();

    var todo = std.ArrayList(*Schedule).init(self.allocator);
    defer todo.deinit();

    if (self.schedules.get(node)) |schedule| {
        try todo.append(schedule);

        while (todo.items.len != 0) {
            const current = todo.pop().?;

            for (current.dependencies) |dep| {
                try todo.append(dep);
            }

            try schedules.insert(0, current);
        }

        return try schedules.toOwnedSlice();
    } else {
        return try self.allocator.alloc(*Schedule, 1);
    }
}
