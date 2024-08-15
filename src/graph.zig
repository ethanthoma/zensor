const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

const zensor = @import("zensor.zig");
const Tensor = zensor.Tensor;
const dtypes = zensor.dtypes;
const Error = zensor.Error;

pub const NodeType = enum {
    Constant,
    Cast,
    Add,
};

pub const OpaqueNode = struct {
    node: *anyopaque,
    evaluateFn: *const anyopaque,

    pub fn init(comptime T: dtypes.DType, comptime S: dtypes.DType, node: *Node(T, S)) OpaqueNode {
        const opaqueNode: *anyopaque = @ptrCast(node);
        const opaqueEvaluateFn: *const anyopaque = @ptrCast(&Node(T, S).evaluate);

        return OpaqueNode{
            .node = opaqueNode,
            .evaluateFn = opaqueEvaluateFn,
        };
    }
};

pub fn Node(comptime T: dtypes.DType, comptime S: dtypes.DType) type {
    return struct {
        const Self = @This();

        pub fn EvaluateFn(comptime Type: dtypes.DType) type {
            return *const fn (*anyopaque) (Error || Allocator.Error)!*Tensor(Type);
        }

        nodeType: NodeType,
        inputs: []const OpaqueNode,
        value: ?*Tensor(S),
        allocator: Allocator,
        evaluateFn: EvaluateFn(S),

        pub fn toOpaqueNode(self: *Self) OpaqueNode {
            const node = OpaqueNode.init(T, S, self);
            return node;
        }

        pub fn deinit(self: *Self) void {
            if (self.value) |v| {
                v.deinit();
                self.allocator.destroy(v);
            }

            self.allocator.free(self.inputs);
        }

        pub fn constant(allocator: Allocator, value: *Tensor(S)) !Self {
            const val = try allocator.create(Tensor(S));
            val.* = try value.clone();

            const inputs = try allocator.alloc(OpaqueNode, 0);

            return Self{
                .nodeType = NodeType.Constant,
                .inputs = inputs,
                .value = val,
                .allocator = allocator,
                // This never gets called
                .evaluateFn = &comptime struct {
                    fn f(self: *anyopaque) !*Tensor(S) {
                        const n: *Self = @alignCast(@ptrCast(self));

                        return n.value.?;
                    }
                }.f,
            };
        }

        pub fn cast(allocator: Allocator, opaqueNode: OpaqueNode) !Self {
            const inputs = try allocator.alloc(OpaqueNode, 1);
            inputs[0] = opaqueNode;

            return Self{
                .nodeType = NodeType.Cast,
                .inputs = inputs,
                .value = null,
                .allocator = allocator,
                .evaluateFn = &comptime struct {
                    fn f(self: *anyopaque) !*Tensor(S) {
                        const n: *Self = @alignCast(@ptrCast(self));

                        const node = n.inputs[0].node;
                        const evaluateFn: EvaluateFn(T) = @ptrCast(n.inputs[0].evaluateFn);
                        const tensor = try evaluateFn(node);
                        const result = try tensor.cast(S);

                        n.value = try n.allocator.create(Tensor(S));
                        n.value.?.* = result;
                        return n.value.?;
                    }
                }.f,
            };
        }

        pub fn add(allocator: Allocator, opaqueNode_a: OpaqueNode, opaqueNode_b: OpaqueNode) !Self {
            const inputs = try allocator.alloc(OpaqueNode, 2);
            inputs[0] = opaqueNode_a;
            inputs[1] = opaqueNode_b;

            return Self{
                .nodeType = NodeType.Add,
                .inputs = inputs,
                .value = null,
                .allocator = allocator,
                .evaluateFn = &comptime struct {
                    fn f(self: *anyopaque) !*Tensor(S) {
                        const n: *Self = @alignCast(@ptrCast(self));

                        const node_a = n.inputs[0].node;
                        const evaluateFn_a: EvaluateFn(T) = @ptrCast(n.inputs[0].evaluateFn);
                        const tensor_a = try evaluateFn_a(node_a);

                        const node_b = n.inputs[1].node;
                        const evaluateFn_b: EvaluateFn(T) = @ptrCast(n.inputs[1].evaluateFn);
                        const tensor_b = try evaluateFn_b(node_b);

                        const result: Tensor(T) = try tensor_a.add(tensor_b);

                        n.value = try n.allocator.create(Tensor(S));
                        n.value.?.* = result;
                        return n.value.?;
                    }
                }.f,
            };
        }

        pub fn evaluate(self: *Self) (Error || Allocator.Error)!*Tensor(S) {
            return self.value orelse try (self.evaluateFn)(self);
        }
    };
}
