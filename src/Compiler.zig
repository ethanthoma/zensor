const std = @import("std");

pub const ast = @import("./compiler/ast.zig");
pub const ir = @import("./compiler/ir.zig");
pub const IRGenerator = @import("./compiler/IRGenerator.zig");
pub const Scheduler = @import("./compiler/Scheduler.zig");

const RuntimeBuffer = @import("./RuntimeBuffer.zig");

const CPU = @import("backend/CPU.zig");
const x86_64 = @import("backend/x86_64.zig");
const jit = @import("backend/jit.zig");

pub const Error = error{
    BufferNotAdded,
};

pub const NodeIndex = enum(u32) {
    _,

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) @TypeOf(writer).Error!void {
        _ = fmt;
        _ = options;

        try writer.print("{}", .{@intFromEnum(self)});
    }
};

pub const NodeManager = struct {
    const Context = struct {
        pub fn hash(self: @This(), key: *const ast.Node) u32 {
            _ = self;
            return ast.Node.hash(key.*);
        }

        pub fn eql(self: @This(), a: *const ast.Node, b: *const ast.Node, b_index: usize) bool {
            _ = self;
            _ = b_index;
            return ast.Node.eql(a.*, b.*);
        }
    };

    allocator: std.mem.Allocator,
    nodes: std.ArrayHashMap(*ast.Node, NodeIndex, Context, true),

    pub fn init(allocator: std.mem.Allocator) NodeManager {
        return .{
            .allocator = allocator,
            .nodes = std.ArrayHashMap(*ast.Node, NodeIndex, Context, true).init(allocator),
        };
    }

    pub fn deinit(self: *NodeManager) void {
        self.nodes.deinit();
    }

    pub fn add(self: *NodeManager, node: ast.Node) !NodeIndex {
        return self.nodes.get(@constCast(&node)) orelse blk: {
            const index: NodeIndex = @enumFromInt(self.nodes.count());

            const node_ptr = try self.allocator.create(ast.Node);
            errdefer self.allocator.destroy(node_ptr);

            node_ptr.* = node;

            try self.nodes.put(node_ptr, index);
            break :blk index;
        };
    }

    pub fn get(self: *const NodeManager, index: NodeIndex) ?*ast.Node {
        const idx: u32 = @intFromEnum(index);
        if (idx < self.nodes.count()) {
            return self.nodes.keys()[idx];
        } else {
            return null;
        }
    }
};

const Compiler = @This();

allocator: std.mem.Allocator,
scheduler: Scheduler,
node_manager: NodeManager,
buffers: std.AutoHashMap(NodeIndex, *RuntimeBuffer),

pub fn init(allocator: std.mem.Allocator) Compiler {
    return .{
        .allocator = allocator,
        .scheduler = Scheduler.init(allocator),
        .node_manager = NodeManager.init(allocator),
        .buffers = std.AutoHashMap(NodeIndex, *RuntimeBuffer).init(allocator),
    };
}

pub fn deinit(self: *Compiler) void {
    self.scheduler.deinit();
    self.node_manager.deinit();
    self.buffers.deinit();
}

pub fn register_buffer(
    self: *Compiler,
    index: NodeIndex,
    buffer: *RuntimeBuffer,
) !void {
    try self.buffers.put(index, buffer);
}

pub fn get_buffer(self: *Compiler, index: NodeIndex) !*RuntimeBuffer {
    return self.buffers.get(index) orelse Error.BufferNotAdded;
}

pub fn mark_for_scheduling(self: *Compiler, index: NodeIndex) !void {
    const node = self.node_manager.get(index).?;
    try self.scheduler.mark_for_scheduling(node);
}

pub fn run(self: *Compiler, index: NodeIndex) !*RuntimeBuffer {
    const node = self.node_manager.get(index).?;
    try self.scheduler.create_schedule(node);

    const schedules = try self.scheduler.fetch_schedules(node);

    var result: *RuntimeBuffer = undefined;
    for (schedules) |schedule| {
        switch (schedule.status) {
            .NotRun => {
                std.log.debug("\n{}", .{schedule});
                schedule.status = .Running;

                const ir_block = try IRGenerator.run(self.allocator, schedule);
                std.log.debug("\n{}", .{ir_block});

                var buffer_map = std.ArrayList([*]u8).init(self.allocator);
                defer buffer_map.deinit();

                for (schedule.global_buffers) |buffer| {
                    try buffer_map.append(buffer.ptr.ptr.ptr);
                }

                const code = try x86_64.generate_kernel(self.allocator, &ir_block);

                std.debug.print("{x:0>2}\n", .{code});

                const mmap = try jit.Code.init(code);
                mmap.run(buffer_map.items.ptr);

                result = schedule.global_buffers[0].ptr;

                schedule.status = .Completed;
            },
            .Completed => blk: {
                for (schedule.global_buffers) |buffer| {
                    if (buffer.idx == 0) {
                        result = buffer.ptr;
                        break :blk;
                    }
                }
                return Error.BufferNotAdded;
            },
            else => unreachable,
        }
    }

    return result;
}
