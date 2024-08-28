const std = @import("std");

const ast = @import("./ast.zig");

const Schedule = struct {
    nodes: []const *const ast.Node,
    output_buffer_id: u32,
    global_buffers: StaticMap(u32, bool),
    reduce_accumulators: StaticMap(*const ast.Node, void),

    pub fn init(
        nodes: []const *const ast.Node,
        output_buffer_id: u32,
        global_buffers: StaticMap(u32, bool),
        reduce_accumulators: StaticMap(*const ast.Node, void),
    ) @This() {
        return .{
            .nodes = nodes,
            .output_buffer_id = output_buffer_id,
            .global_buffers = global_buffers,
            .reduce_accumulators = reduce_accumulators,
        };
    }
};

const Scheduler = @This();

pub fn init() Scheduler {
    return .{};
}

fn ast_node_eql(lhs: *const ast.Node, rhs: *const ast.Node) bool {
    return lhs == rhs;
}

pub fn run(comptime node: *const ast.Node) []const *const ast.Node {
    comptime {
        var visited = StaticArrayList(*const ast.Node).init(ast_node_eql);

        var order = StaticArrayList(*const ast.Node).init(ast_node_eql);

        topological_sort(node, &visited, &order);

        const final_order = order.buffer;

        return final_order;
    }
}

fn topological_sort(
    comptime node: *const ast.Node,
    comptime visited: *StaticArrayList(*const ast.Node),
    comptime order: *StaticArrayList(*const ast.Node),
) void {
    comptime {
        if (visited.contains(node)) {
            return;
        }

        visited.append(node);

        const inputs = @field(node.input, @tagName(node.op));

        if (@typeInfo(@TypeOf(inputs)) == .Array) {
            for (inputs) |input| {
                topological_sort(input, visited, order);
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
