const std = @import("std");
const posix = std.posix;
const mem = std.mem;
const heap = std.heap;

const RuntimeBuffer = @import("../RuntimeBuffer.zig");

pub const Code = struct {
    mmap_region: []const u8,

    pub fn init(code: []const u8) !Code {
        const len = (code.len / heap.page_size_min + 1) * heap.page_size_min;

        var mmap_region = try posix.mmap(
            null,
            len,
            posix.PROT.READ | posix.PROT.WRITE,
            .{
                .TYPE = .PRIVATE,
                .ANONYMOUS = true,
            },
            -1,
            0,
        );
        @memcpy(mmap_region[0..code.len], code);

        try posix.mprotect(mmap_region, posix.PROT.READ | posix.PROT.EXEC);

        return .{ .mmap_region = mmap_region };
    }

    pub fn deinit(self: *const Code) void {
        posix.munmap(self.mmap_region);
    }

    pub fn run(self: *const Code, buffers: [*][*]u8) void {
        const f: *const fn ([*][*]u8) callconv(.C) void = @alignCast(@ptrCast(self.mmap_region.ptr));
        @call(.auto, f, .{buffers});
    }
};
