const std = @import("std");

const ast = @import("./ast.zig");

pub const Schedule = struct {
    nodes: []const *const ast.Node,
    output_buffer_id: u32,
    global_buffers: []const ast.BufferID,

    pub fn init(
        nodes: []const *const ast.Node,
        output_buffer_id: u32,
        global_buffers: []const ast.BufferID,
    ) @This() {
        return .{
            .nodes = nodes,
            .output_buffer_id = output_buffer_id,
            .global_buffers = global_buffers,
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
        try writer.print("{}, ", .{self.output_buffer_id});
        try writer.print("[", .{});
        for (self.global_buffers, 0..) |buffer_id, i| {
            try writer.print("{any}", .{buffer_id});
            if (i < self.global_buffers.len - 1) {
                try writer.print(", ", .{});
            }
        }
        try writer.print("]}}", .{});
    }
};

const Scheduler = @This();

pub fn init() Scheduler {
    return .{};
}

fn ast_node_eql(lhs: *const ast.Node, rhs: *const ast.Node) bool {
    return lhs == rhs;
}

fn buffer_id_eql(lhs: ast.BufferID, rhs: ast.BufferID) bool {
    return lhs == rhs;
}

// TODO: split ast into multiple kernels
pub fn run(comptime node: *const ast.Node) []const Schedule {
    comptime {
        var visited = StaticArrayList(*const ast.Node).init(ast_node_eql);
        var order = StaticArrayList(*const ast.Node).init(ast_node_eql);
        var buffers = StaticArrayList(ast.BufferID).init(buffer_id_eql);

        // TODO: I need to generate a buffer_id, idk how to tho
        const buffer_id = 12345;

        const store_node = &ast.Node.init(
            .Store,
            .{ .buffer_id = buffer_id },
            [_]*const ast.Node{node},
            node.view,
            node.dtype,
        );

        topological_sort(store_node, &visited, &order, &buffers);

        const schedule = Schedule.init(
            order.buffer,
            buffer_id,
            buffers.buffer,
        );

        const schedules: []const Schedule = &[_]Schedule{schedule};

        return schedules;
    }
}

fn topological_sort(
    comptime node: *const ast.Node,
    comptime visited: *StaticArrayList(*const ast.Node),
    comptime order: *StaticArrayList(*const ast.Node),
    comptime buffers: *StaticArrayList(ast.BufferID),
) void {
    comptime {
        if (visited.contains(node)) {
            return;
        }

        switch (node.op) {
            .Load => {
                buffers.append(node.arg.Load.buffer_id);
            },
            .Store => {
                buffers.append(node.arg.Store.buffer_id);
            },
            else => {},
        }

        visited.append(node);

        const inputs = @field(node.input, @tagName(node.op));

        if (@typeInfo(@TypeOf(inputs)) == .Array) {
            for (inputs) |input| {
                topological_sort(input, visited, order, buffers);
            }
        }

        order.append(node);
    }
}

fn StaticArrayList(comptime T: type) type {
    return struct {
        buffer: []const T = &empty_buffer,
        eql: *const fn (T, T) bool,

        const empty_buffer = [_]T{};

        pub fn init(comptime eql: *const fn (T, T) bool) @This() {
            comptime {
                return .{
                    .eql = eql,
                };
            }
        }

        pub fn append(comptime self: *@This(), comptime value: T) void {
            comptime {
                var buffer: [self.buffer.len + 1]T = undefined;
                @memcpy(buffer[0..self.buffer.len], self.buffer);
                buffer[self.buffer.len] = value;

                const final_buffer = buffer;
                self.buffer = &final_buffer;
            }
        }

        pub fn contains(comptime self: *const @This(), comptime value: T) bool {
            for (self.buffer) |item| {
                if (self.eql(item, value)) return true;
            }

            return false;
        }
    };
}

fn StaticMap(comptime K: type, comptime V: type) type {
    return struct {
        keys: []const K = &empty_keys,
        values: []const V = &empty_values,
        eql: *const fn (K, K) bool,

        const empty_keys = [_]K{};
        const empty_values = [_]V{};

        pub fn init(comptime eql: *const fn (K, K) bool) @This() {
            comptime {
                return .{
                    .eql = eql,
                };
            }
        }

        pub fn put(comptime self: *@This(), comptime key: K, comptime value: V) void {
            comptime {
                if (self.has(key)) {
                    @compileError(std.fmt.comptimePrint("Key {} is already in map", .{key}));
                }

                var keys: [self.keys.len + 1]K = undefined;
                @memcpy(keys[0..self.keys.len], self.keys);
                keys[self.keys.len] = value;

                const final_keys = keys;
                self.keys = &final_keys;

                var values: [self.values.len + 1]K = undefined;
                @memcpy(values[0..self.values.len], self.values);
                values[self.values.len] = value;

                const final_values = values;
                self.values = &final_values;
            }
        }

        pub fn has(comptime self: *const @This(), comptime key: K) bool {
            comptime {
                for (self.keys) |contained_key| {
                    if (self.eql(key, contained_key)) {
                        return true;
                    }
                }
                return false;
            }
        }
    };
}
