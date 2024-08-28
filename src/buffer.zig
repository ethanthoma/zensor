const std = @import("std");

const dtypes = @import("./dtypes.zig");

pub const RuntimeBuffer = struct {
    ptr: []const u8,
    len: u32,
    dtype: dtypes.DataType,
    shape: []const u32,
};

pub const Numpy = struct {
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !RuntimeBuffer {
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var reader = file.reader();

        var magic: [6]u8 = undefined;
        try reader.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "\x93NUMPY")) {
            return error.InvalidNumpyFormat;
        }

        const major_version = try reader.readByte();
        const minor_version = try reader.readByte();
        if (major_version != 1 or minor_version != 0) {
            return error.UnsupportedNumpyVersion;
        }

        const header_len = try reader.readInt(u16, std.builtin.Endian.little);

        const header = try allocator.alloc(u8, header_len);
        defer allocator.free(header);
        try reader.readNoEof(header);

        var dtype: []const u8 = undefined;
        var fortran_order: bool = false;
        var shape: []u32 = undefined;

        var it = std.mem.splitScalar(u8, header, ',');
        while (it.next()) |item| {
            var kv = std.mem.splitScalar(u8, item, ':');

            if (kv.next()) |key| {
                if (std.mem.indexOf(u8, key, "descr") != null) {
                    dtype = std.mem.trim(u8, kv.next() orelse return error.InvalidHeader, " '");
                } else if (std.mem.indexOf(u8, key, "fortran_order") != null) {
                    fortran_order = if (std.mem.indexOf(u8, kv.next() orelse return error.InvalidHeader, "True") != null) true else false;
                } else if (std.mem.indexOf(u8, key, "shape") != null) {
                    const shape_str = std.mem.trim(u8, kv.next() orelse return error.InvalidHeader, " ()");
                    var shape_it = std.mem.splitScalar(u8, shape_str, ' ');
                    var shape_list = std.ArrayList(u32).init(allocator);
                    defer shape_list.deinit();
                    while (shape_it.next()) |dim| {
                        try shape_list.append(try std.fmt.parseInt(u32, dim, 10));
                    }
                    shape = try shape_list.toOwnedSlice();
                }
            }
        }

        const datatype = dtypes.FromNumpy(dtype) orelse return error.Invalid;

        var size: u32 = 1;
        for (shape) |dim| {
            size *= dim;
        }

        const data: []u8 = try allocator.alloc(u8, size * datatype.bits / 8);

        try reader.readNoEof(data);

        return RuntimeBuffer{
            .ptr = data,
            .len = size,
            .dtype = datatype,
            .shape = shape,
        };
    }
};
