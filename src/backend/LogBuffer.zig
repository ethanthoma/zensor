const std = @import("std");

const LogBuffer = @This();

messages: std.ArrayList([]const u8),
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator) LogBuffer {
    return .{
        .messages = std.ArrayList([]const u8).init(allocator),
        .allocator = allocator,
    };
}

pub fn deinit(self: *LogBuffer) void {
    for (self.messages.items) |msg| {
        self.allocator.free(msg);
    }
    self.messages.deinit();
}

pub fn log(self: *LogBuffer, comptime fmt: []const u8, args: anytype) !void {
    const msg = try std.fmt.allocPrint(self.allocator, fmt, args);
    try self.messages.append(msg);
}

pub fn dump(self: LogBuffer) !void {
    const merge = try std.mem.join(self.allocator, "", self.messages.items);

    var iter = std.mem.splitScalar(u8, merge, '\n');
    while (iter.next()) |msg| {
        if (msg.len > 0) std.log.debug("{s}", .{msg});
    }
}
