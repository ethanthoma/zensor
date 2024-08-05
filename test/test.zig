const std = @import("std");
const testing = std.testing;

const Tensor = @import("zensor").Tensor;
const Error = @import("zensor").Error;

test "empty" {
    {
        const T = u64;
        const shape = 42;
        const tensor = try Tensor(T).empty(testing.allocator, shape);
        defer tensor.deinit();

        const result: []const usize = tensor.shape;
        const expected = ([_]usize{42})[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }
    {
        const T = u64;
        const shape = .{ 1, 2, 3 };
        const tensor = try Tensor(T).empty(testing.allocator, shape);
        defer tensor.deinit();

        const result: []const usize = tensor.shape;
        const expected = ([_]usize{ 1, 2, 3 })[0..];

        try testing.expectEqualSlices(usize, expected, result);
    }
}

test "clone" {
    const allocator = testing.allocator;

    {
        const T = u64;
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
        const T = u64;
        var tensor = try Tensor(T).arange(allocator, 5, 13);
        defer tensor.deinit();

        try tensor.reshape(.{ 2, 4 });

        for (tensor.data(), 0..) |elem, idx| {
            const v: u64 = 5 + idx;
            try testing.expectEqual(v, elem);
        }
    }
}

test "fromOwnedSlice" {
    const allocator = testing.allocator;

    {
        const T = u64;
        const slice: []const T = &.{ 1, 2, 3, 4, 5 };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]usize{5})[0..];
        try testing.expectEqualSlices(usize, expectedShape, tensor.shape);

        const actualData: []const T = @alignCast(@ptrCast(tensor.data()));
        try testing.expectEqualSlices(T, slice, actualData);
    }
    {
        const T = u64;
        const slice: []const T = &.{};
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]usize{0})[0..];
        try testing.expectEqualSlices(usize, expectedShape, tensor.shape);

        const actualData: []const T = @alignCast(@ptrCast(tensor.data()));
        try testing.expectEqualSlices(T, slice, actualData);
    }
    {
        const T = u64;
        const slice: []const T = &.{ 1, 2, 3, 4, 5 };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]usize{5})[0..];
        try testing.expectEqualSlices(usize, expectedShape, tensor.shape);

        const actualData: []const T = @alignCast(@ptrCast(tensor.data()));
        try testing.expectEqualSlices(T, slice, actualData);
    }
    {
        const T = u64;
        const slice: []const []const T = &.{ &.{ 1, 2, 3 }, &.{ 4, 5, 6 } };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]usize{ 2, 3 })[0..];
        try testing.expectEqualSlices(usize, expectedShape, tensor.shape);

        const actualData: []const T = @alignCast(@ptrCast(tensor.data()));
        for (slice, 0..) |row, i| {
            for (row, 0..) |_, j| {
                const idx = i * 3 + j;
                try testing.expectEqual(slice[i][j], actualData[idx]);
            }
        }
    }
    {
        const T = u64;
        const slice: []const []const []const T = &.{ &.{ &.{ 1, 2 }, &.{ 3, 4 } }, &.{ &.{ 5, 6 }, &.{ 7, 8 } } };
        const tensor = try Tensor(T).fromOwnedSlice(allocator, slice);
        defer tensor.deinit();

        const expectedShape = ([_]usize{ 2, 2, 2 })[0..];
        try testing.expectEqualSlices(usize, expectedShape, tensor.shape);

        const actualData: []const T = @alignCast(@ptrCast(tensor.data()));
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
        const T = u64;
        var tensor1: Tensor(T) = try Tensor(u64).arange(allocator, 5, 13);
        defer tensor1.deinit();

        try tensor1.reshape(.{ 2, 4 });

        var tensor2 = try Tensor(T).full(allocator, .{ 3, 2, 4 }, 2);
        defer tensor2.deinit();

        const result = try tensor1.add(&tensor2);
        defer result.deinit();

        for (result.data(), 0..) |elem, idx| {
            const v: T = (idx % 8) + 7;
            try testing.expectEqual(v, elem);
        }
    }
    {
        const T = u64;
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
        const T = u64;
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
        const T = u64;
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
        const T = u64;
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
        const T = u64;
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
        const T = u64;
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
        const T = u64;
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
        const T = u64;
        var self: Tensor(T) = try Tensor(u64).arange(allocator, 5, 13);
        defer self.deinit();

        const other: T = 2;

        const result = try self.add(other);
        defer result.deinit();

        for (result.data(), 0..) |elem, idx| {
            const v: T = idx + 5 + other;
            try testing.expectEqual(v, elem);
        }
    }
}

test {
    testing.refAllDecls(@This());
}
