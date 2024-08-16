const std = @import("std");
const testing = std.testing;

const zensor = @import("zensor");
const Tensor = zensor.Tensor;
const Error = zensor.Error;
const dtypes = zensor.dtypes;

test "empty" {
    {
        const T = dtypes.int32;
        const shape = 42;
        const tensor = try Tensor(T).empty(testing.allocator, shape);
        defer tensor.deinit();

        const result: []const u32 = tensor.shape;

        try testing.expectEqual(shape, result[0]);
    }
    {
        const T = dtypes.int32;
        const shape = .{ 1, 2, 3 };
        const tensor = try Tensor(T).empty(testing.allocator, shape);
        defer tensor.deinit();

        const result: []const u32 = tensor.shape;
        const expected = ([_]u32{ 1, 2, 3 })[0..];

        try testing.expectEqualSlices(u32, expected, result);
    }
}

test "clone" {
    const allocator = testing.allocator;

    {
        const T = dtypes.int32;
        var tensor1 = try Tensor(T).arange(allocator, 5, 13);
        defer tensor1.deinit();

        const tensor2 = try tensor1.clone();
        defer tensor2.deinit();

        for (tensor1.data(), tensor2.data()) |elem1, elem2| {
            try testing.expectEqual(elem1, elem2);
        }
    }
}

test "reshape" {
    const allocator = testing.allocator;

    {
        const T = dtypes.int32;
        var tensor = try Tensor(T).arange(allocator, 5, 13);
        defer tensor.deinit();

        try tensor.reshape(.{ 2, 4 });

        for (tensor.data(), 0..) |elem, idx| {
            const v: T.kind = 5 + @as(T.kind, @intCast(idx));
            try testing.expectEqual(v, elem);
        }
    }
}

test "fromOwnedSlice" {
    const allocator = testing.allocator;

    {
        const T = dtypes.int32;
        const slice: []const T.kind = &.{ 1, 2, 3, 4, 5 };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]u32{5})[0..];
        try testing.expectEqualSlices(u32, expectedShape, tensor.shape);

        const actualData: []const T.kind = @alignCast(@ptrCast(tensor.data()));
        try testing.expectEqualSlices(T.kind, slice, actualData);
    }
    {
        const T = dtypes.int32;
        const slice: []const T.kind = &.{};
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]u32{0})[0..];
        try testing.expectEqualSlices(u32, expectedShape, tensor.shape);

        const actualData: []const T.kind = @alignCast(@ptrCast(tensor.data()));
        try testing.expectEqualSlices(T.kind, slice, actualData);
    }
    {
        const T = dtypes.int32;
        const slice: []const T.kind = &.{ 1, 2, 3, 4, 5 };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]u32{5})[0..];
        try testing.expectEqualSlices(u32, expectedShape, tensor.shape);

        const actualData: []const T.kind = @alignCast(@ptrCast(tensor.data()));
        try testing.expectEqualSlices(T.kind, slice, actualData);
    }
    {
        const T = dtypes.int32;
        const slice: []const []const T.kind = &.{ &.{ 1, 2, 3 }, &.{ 4, 5, 6 } };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]u32{ 2, 3 })[0..];
        try testing.expectEqualSlices(u32, expectedShape, tensor.shape);

        const actualData: []const T.kind = @alignCast(@ptrCast(tensor.data()));
        for (slice, 0..) |row, i| {
            for (row, 0..) |_, j| {
                const idx = i * 3 + j;
                try testing.expectEqual(slice[i][j], actualData[idx]);
            }
        }
    }
    {
        const T = dtypes.int32;
        const slice: []const []const []const T.kind = &.{ &.{ &.{ 1, 2 }, &.{ 3, 4 } }, &.{ &.{ 5, 6 }, &.{ 7, 8 } } };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]u32{ 2, 2, 2 })[0..];
        try testing.expectEqualSlices(u32, expectedShape, tensor.shape);

        const actualData: []const i32 = @alignCast(@ptrCast(tensor.data()));
        for (slice, 0..) |plane, i| {
            for (plane, 0..) |row, j| {
                for (row, 0..) |_, k| {
                    const idx = i * 4 + j * 2 + k;
                    try testing.expectEqual(slice[i][j][k], actualData[idx]);
                }
            }
        }
    }
}

test "add" {
    const allocator = testing.allocator;

    {
        const T = dtypes.int32;
        var tensor1: Tensor(T) = try Tensor(T).arange(allocator, 5, 13);
        defer tensor1.deinit();

        try tensor1.reshape(.{ 2, 4 });

        var tensor2 = try Tensor(T).full(allocator, .{ 3, 2, 4 }, 2);
        defer tensor2.deinit();

        const result = try tensor1.add(&tensor2);
        defer result.deinit();

        for (result.data(), 0..) |elem, idx| {
            const v: T.kind = @as(T.kind, @intCast(idx % 8)) + 7;
            try testing.expectEqual(v, elem);
        }
    }
    {
        const T = dtypes.int32;
        const shape1 = .{ 5, 4 };
        const shape2 = .{1};

        var tensor1 = try Tensor(T).ones(allocator, shape1);
        defer tensor1.deinit();

        var tensor2 = try Tensor(T).ones(allocator, shape2);
        defer tensor2.deinit();

        const result = try tensor1.add(&tensor2);
        defer result.deinit();

        for (result.data()) |elem| {
            try testing.expectEqual(2, elem);
        }
    }
    {
        const T = dtypes.int32;
        const shape1 = .{ 5, 4 };
        const shape2 = .{4};

        var tensor1 = try Tensor(T).ones(allocator, shape1);
        defer tensor1.deinit();

        var tensor2 = try Tensor(T).ones(allocator, shape2);
        defer tensor2.deinit();

        const result = try tensor1.add(&tensor2);
        defer result.deinit();

        for (result.data()) |elem| {
            try testing.expectEqual(2, elem);
        }
    }
    {
        const T = dtypes.int32;
        const shape1 = .{ 15, 3, 5 };
        const shape2 = .{ 15, 1, 5 };

        var tensor1 = try Tensor(T).ones(allocator, shape1);
        defer tensor1.deinit();

        var tensor2 = try Tensor(T).ones(allocator, shape2);
        defer tensor2.deinit();

        const result = try tensor1.add(&tensor2);
        defer result.deinit();

        for (result.data()) |elem| {
            try testing.expectEqual(2, elem);
        }
    }
    {
        const T = dtypes.int32;
        const shape1 = .{ 15, 3, 5 };
        const shape2 = .{ 3, 5 };

        var tensor1 = try Tensor(T).ones(allocator, shape1);
        defer tensor1.deinit();

        var tensor2 = try Tensor(T).ones(allocator, shape2);
        defer tensor2.deinit();

        const result = try tensor1.add(&tensor2);
        defer result.deinit();

        for (result.data()) |elem| {
            try testing.expectEqual(2, elem);
        }
    }
    {
        const T = dtypes.int32;
        const shape1 = .{ 15, 3, 5 };
        const shape2 = .{ 3, 1 };

        var tensor1 = try Tensor(T).ones(allocator, shape1);
        defer tensor1.deinit();

        var tensor2 = try Tensor(T).ones(allocator, shape2);
        defer tensor2.deinit();

        const result = try tensor1.add(&tensor2);
        defer result.deinit();

        for (result.data()) |elem| {
            try testing.expectEqual(2, elem);
        }
    }
    {
        const T = dtypes.int32;
        const shape1 = .{3};
        const shape2 = .{4};

        var tensor1 = try Tensor(T).ones(allocator, shape1);
        defer tensor1.deinit();

        var tensor2 = try Tensor(T).ones(allocator, shape2);
        defer tensor2.deinit();

        const result = tensor1.add(&tensor2);
        const expected = Error.CannotBroadcast;

        try testing.expectError(expected, result);
    }
    {
        const T = dtypes.int32;
        const shape1 = .{ 2, 1 };
        const shape2 = .{ 8, 4, 3 };

        var tensor1 = try Tensor(T).ones(allocator, shape1);
        defer tensor1.deinit();

        var tensor2 = try Tensor(T).ones(allocator, shape2);
        defer tensor2.deinit();

        const result = tensor1.add(&tensor2);
        const expected = Error.CannotBroadcast;

        try testing.expectError(expected, result);
    }
    {
        const T = dtypes.int32;
        var self = try Tensor(T).arange(allocator, 5, 13);
        defer self.deinit();

        const other: T.kind = 2;

        const result = try self.add(other);
        defer result.deinit();

        for (result.data(), 0..) |elem, idx| {
            const v: T.kind = @as(T.kind, @intCast(idx)) + 5 + other;
            try testing.expectEqual(v, elem);
        }
    }
}

test {
    testing.refAllDecls(@This());
}
