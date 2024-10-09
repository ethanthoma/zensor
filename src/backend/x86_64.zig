const std = @import("std");
const mem = std.mem;
const fmt = std.fmt;
const io = std.io;
const assert = std.debug.assert;

const ir = @import("../compiler/ir.zig");

// rbp: [*][*]u8
// rdi: function arg 1
// rsi: function arg 2

const Context = struct {
    block: *const ir.Block,
    cursor: usize = 0,
    code: *std.ArrayList(u8),
    writer: *std.ArrayList(u8).Writer,
    store: *ValueStore,
    labels: *std.AutoHashMap(ir.Step, usize),
};

const ReturnRegister = enum { Rax };

const StackRegister = enum { Rsp };

const ParamRegister = enum { Rbp };

const FunctionCallingRegister = enum { Rdi, Rsi };

const StaticRegister = enum {
    Rax,
    Rcx,
    Rdx,
    Rdi,
    Rsi,

    pub fn to_generic(self: @This()) Register {
        inline for (std.meta.fields(Register)) |field| {
            if (mem.eql(u8, @tagName(self), field.name)) {
                return @enumFromInt(field.value);
            }
        } else {
            unreachable;
        }
    }
};

const TempRegister = enum {
    R8,
    R9,
    R10,
    R11,

    pub fn to_generic(self: @This()) Register {
        inline for (std.meta.fields(Register)) |field| {
            if (mem.eql(u8, @tagName(self), field.name)) {
                return @enumFromInt(field.value);
            }
        } else {
            unreachable;
        }
    }
};

const Register = blk: {
    const register_list = [_]type{
        ReturnRegister,
        StackRegister,
        ParamRegister,
        FunctionCallingRegister,
        StaticRegister,
        TempRegister,
    };

    var length = 0;
    for (register_list) |T| {
        length += std.meta.fields(T).len;
    }

    var idx = 0;
    var fields: [length]std.builtin.Type.EnumField = undefined;
    for (register_list) |T| {
        loop: for (std.meta.fields(T)) |field| {
            // check if name already in enum
            for (0..idx) |i| {
                if (std.mem.eql(u8, fields[i].name, field.name)) {
                    continue :loop;
                }
            }

            fields[idx] = .{
                .name = field.name,
                .value = idx,
            };

            idx += 1;
        }
    }

    break :blk @Type(.{
        .Enum = .{
            .tag_type = usize,
            .fields = fields[0..idx],
            .decls = &[_]std.builtin.Type.Declaration{}, // has to be empty
            .is_exhaustive = true,
        },
    });
};

const ValueStore = struct {
    map: std.AutoHashMap(ir.Step, Location),
    static_list: std.ArrayList(StaticRegister),
    temp_list: std.ArrayList(TempRegister),
    temp_steps: std.AutoHashMap(TempRegister, ir.Step),
    consts: std.AutoHashMap(ir.Step, []const u8),
    allocator: mem.Allocator,
    writer: *std.ArrayList(u8).Writer,
    offset: i32 = 0,

    pub fn init(allocator: mem.Allocator, writer: anytype) !ValueStore {
        var value_store = ValueStore{
            .map = std.AutoHashMap(ir.Step, Location).init(allocator),
            .static_list = std.ArrayList(StaticRegister).init(allocator),
            .temp_list = std.ArrayList(TempRegister).init(allocator),
            .temp_steps = std.AutoHashMap(TempRegister, ir.Step).init(allocator),
            .consts = std.AutoHashMap(ir.Step, []const u8).init(allocator),
            .allocator = allocator,
            .writer = writer,
        };

        try value_store.static_list.appendSlice(std.meta.tags(StaticRegister));
        try value_store.temp_list.appendSlice(std.meta.tags(TempRegister));

        return value_store;
    }

    pub const Location = union(enum) {
        Temp: TempRegister,
        Static: StaticRegister,
        Stack: i32,
    };

    // guarantees that the value(s) are still accessable at all times
    // value should be in little endian
    pub fn write_value_static(self: *ValueStore, step: ir.Step, value: [4]u8) !void {
        if (self.static_list.popOrNull()) |reg| {
            const loc = Location{ .Static = reg };

            try write_value_to_reg(reg.to_generic(), value, self.writer);

            try self.map.put(step, loc);
        } else {
            const loc = Location{
                .Stack = -self.offset,
            };

            self.offset += 4;

            try push_to_stack(value, self.writer);

            try self.map.put(step, loc);
        }
    }

    fn push_to_stack(value: [4]u8, writer: anytype) !void {
        std.log.debug("push {x}", .{std.mem.bytesToValue(u32, &value)});

        try writer.writeByte(0x68);
        try writer.writeAll(&value);
    }

    // no guarantee that the value is still stored after some time
    // use for temp vars
    // value should be in little endian
    pub fn write_value_temp(self: *ValueStore, step: ir.Step, value: [4]u8) !void {
        const reg = self.temp_list.pop();

        if (self.temp_steps.get(reg)) |prev| {
            self.temp_steps.remove(reg);
            self.map.remove(prev);
        }

        const loc = Location{ .Temp = reg };

        try write_value_to_reg(reg.toGeneric(), value, self.writer);

        try self.map.put(step, loc);

        try self.temp_list.append(reg);
    }

    fn write_value_to_reg(reg: Register, value: [4]u8, writer: anytype) !void {
        if (mem.bytesToValue(u32, &value) == 0) {
            std.log.debug("xor {s}, {s}", .{ @tagName(reg), @tagName(reg) });

            switch (reg) {
                .Rax, .Rcx, .Rdx, .Rdi, .Rsi => try writer.writeByte(0x48),
                .R8, .R9, .R10, .R11 => try writer.writeByte(0x4d),
                .Rsp, .Rbp => unreachable,
            }

            try writer.writeByte(0x31);

            switch (reg) {
                .Rax => try writer.writeByte(0xc0),
                .Rcx => try writer.writeByte(0xc9),
                .Rdx => try writer.writeByte(0xd2),
                .Rdi => try writer.writeByte(0xff),
                .Rsi => try writer.writeByte(0xf6),
                .R8 => try writer.writeByte(0xc0),
                .R9 => try writer.writeByte(0xc9),
                .R10 => try writer.writeByte(0xd2),
                .R11 => try writer.writeByte(0xdb),
                .Rsp, .Rbp => unreachable,
            }
        } else {
            std.log.debug("mov {s}, {x}", .{ @tagName(reg), std.mem.bytesToValue(u32, &value) });

            switch (reg) {
                .Rax, .Rcx, .Rdx, .Rdi, .Rsi => try writer.writeByte(0x48),
                .R8, .R9, .R10, .R11 => try writer.writeByte(0x49),
                .Rsp, .Rbp => unreachable,
            }

            try writer.writeByte(0xc7);

            switch (reg) {
                .Rax => try writer.writeByte(0xc0),
                .Rcx => try writer.writeByte(0xc1),
                .Rdx => try writer.writeByte(0xc2),
                .Rdi => try writer.writeByte(0xc7),
                .Rsi => try writer.writeByte(0xc6),
                .R8 => try writer.writeByte(0xc0),
                .R9 => try writer.writeByte(0xc1),
                .R10 => try writer.writeByte(0xc2),
                .R11 => try writer.writeByte(0xc3),
                .Rsp, .Rbp => unreachable,
            }

            try writer.writeAll(&value);
        }
    }

    // loads 4 bytes from a buffer
    // stores value in a temporary register
    pub fn load_from_buffer(self: *ValueStore, step: ir.Step, buffer_idx: u8) !void {
        const reg = self.temp_list.pop();

        const loc = Location{ .Temp = reg };

        try load_value_from_register(reg.to_generic(), .Rbp, buffer_idx, self.writer);

        try self.map.put(step, loc);

        try self.temp_list.append(reg);
    }

    fn load_value_from_register(dest: Register, source: Register, idx: u8, writer: anytype) !void {
        std.log.debug("mov {s}, qword ptr [{s} + {d}]", .{ @tagName(dest), @tagName(source), idx });

        // REX prefix + Opcode
        switch (dest) {
            .Rax, .Rcx, .Rdx, .Rdi, .Rsi => try writer.writeByte(0x48),
            .R8, .R9, .R10, .R11 => try writer.writeByte(0x4c),
            .Rsp, .Rbp => unreachable,
        }

        try writer.writeByte(0x8b);

        try writer.writeByte(modrm(dest, source));

        // SIB byte for [RSP] with no index
        if (source == .Rsp) {
            try writer.writeByte(0x24);
        }

        try writer.writeByte(idx);
    }

    fn modrm(dest: Register, source: Register) u8 {
        const mod: u8 = 0b01;

        const reg: u8 = switch (dest) {
            .Rax => 0b000,
            .Rcx => 0b001,
            .Rdx => 0b010,
            .Rsi => 0b110,
            .Rdi => 0b111,

            .R8 => 0b000,
            .R9 => 0b001,
            .R10 => 0b010,
            .R11 => 0b011,

            .Rsp, .Rbp => unreachable,
        };

        const rm: u8 = switch (source) {
            .Rax => 0b000,
            .Rcx => 0b001,
            .Rdx => 0b010,
            .Rsp => 0b100,
            .Rsi => 0b110,
            .Rdi => 0b111,

            .R8 => 0b000,
            .R9 => 0b001,
            .R10 => 0b010,
            .R11 => 0b011,

            .Rbp => 0b101,
        };

        return mod << 6 | reg << 3 | rm;
    }

    pub fn write_const(self: *ValueStore, step: ir.Step, value: [4]u8) !void {
        const buffer = try self.allocator.dupe(u8, &value);
        try self.consts.put(step, buffer);
    }

    pub fn copy_from_static(self: *ValueStore, dest: ir.Step, source: ir.Step) !void {
        if (self.map.get(source)) |source_loc| {
            if (self.static_list.popOrNull()) |dest_reg| {
                const loc = Location{ .Static = dest_reg };
                try self.map.put(dest, loc);

                switch (source_loc) {
                    .Temp => |source_reg| {
                        try load_value_from_register(dest_reg.to_generic(), source_reg.to_generic(), 0, self.writer);
                    },
                    .Static => |source_reg| {
                        try load_value_from_register(dest_reg.to_generic(), source_reg.to_generic(), 0, self.writer);
                    },
                    .Stack => |offset| {
                        // pop dest
                        std.log.debug("pop {s}", .{@tagName(dest_reg)});
                        _ = offset;
                        std.debug.panic("copy_from_static: register from stack not implemented", .{});
                    },
                }
            } else {
                std.debug.panic("copy_from_static: stack from anything not implemented", .{});
            }
        } else {
            if (self.consts.get(source)) |const_value| {
                if (self.static_list.popOrNull()) |dest_reg| {
                    const loc = Location{ .Static = dest_reg };
                    try self.map.put(dest, loc);

                    const value: [4]u8 = const_value.ptr[0..4].*;
                    try write_value_to_reg(dest_reg.to_generic(), value, self.writer);
                } else {
                    const loc = Location{
                        .Stack = -self.offset,
                    };

                    self.offset += 4;

                    const value: [4]u8 = const_value.ptr[0..4].*;
                    try push_to_stack(value, self.writer);

                    try self.map.put(dest, loc);
                }
            } else {
                std.debug.panic("copy_from_static: source is a deallocated temp value", .{});
            }
        }
    }

    pub fn inc(self: *ValueStore, step: ir.Step) !void {
        if (self.map.get(step)) |loc| {
            switch (loc) {
                .Temp => |reg| {
                    try inc_register(reg.to_generic(), self.writer);
                },
                .Static => |reg| {
                    try inc_register(reg.to_generic(), self.writer);
                },
                .Stack => |offset| {
                    // pop dest
                    _ = offset;
                    std.debug.panic("inc: stack inc not implemented", .{});
                },
            }
        } else {
            if (self.consts.get(step)) |_| {
                std.debug.panic("inc: cannot inc on a const value", .{});
            } else {
                std.debug.panic("inc: source is a deallocated temp value", .{});
            }
        }
    }

    fn inc_register(reg: Register, writer: anytype) !void {
        std.log.debug("inc {s}", .{@tagName(reg)});
        switch (reg) {
            .Rax, .Rcx, .Rdx, .Rdi, .Rsi => try writer.writeByte(0x48),
            .R8, .R9, .R10, .R11 => try writer.writeByte(0x4d),
            .Rsp, .Rbp => unreachable,
        }

        try writer.writeByte(0xff);

        switch (reg) {
            .Rax => try writer.writeByte(0xc0),
            .Rcx => try writer.writeByte(0xc1),
            .Rdx => try writer.writeByte(0xc2),
            .Rdi => try writer.writeByte(0xc7),
            .Rsi => try writer.writeByte(0xc6),
            .R8 => try writer.writeByte(0xc0),
            .R9 => try writer.writeByte(0xc1),
            .R10 => try writer.writeByte(0xc2),
            .R11 => try writer.writeByte(0xc3),
            .Rsp, .Rbp => unreachable,
        }
    }

    pub fn cmp(self: *ValueStore, dest: ir.Step, source: ir.Step) !void {
        const dest_reg = if (self.map.get(dest)) |loc| blk: {
            switch (loc) {
                .Temp => |reg| {
                    break :blk reg.to_generic();
                },
                .Static => |reg| {
                    break :blk reg.to_generic();
                },
                .Stack => |offset| {
                    _ = offset;
                    std.debug.panic("cmp: register and stack comparison not implemented", .{});
                },
            }
        } else {
            // TODO: we can just swap it?
            if (self.consts.get(source)) |_| {
                std.debug.panic("cmp: const and register comparison not allowed", .{});
            } else {
                std.debug.panic("cmp: source is a deallocated temp value", .{});
            }
        };

        const source_reg = if (self.map.get(source)) |loc| blk: {
            switch (loc) {
                .Temp => |reg| {
                    break :blk reg.to_generic();
                },
                .Static => |reg| {
                    break :blk reg.to_generic();
                },
                .Stack => |offset| {
                    _ = offset;
                    std.debug.panic("cmp: register and stack comparison not implemented", .{});
                },
            }
        } else {
            if (self.consts.get(source)) |const_value| {
                const value: [4]u8 = const_value.ptr[0..4].*;
                try cmp_register_with_const(dest_reg, value, self.writer);
                return;
            } else {
                std.debug.panic("cmp: source is a deallocated temp value", .{});
            }
        };

        try cmp_registers(dest_reg, source_reg, self.writer);
    }

    fn cmp_registers(dest: Register, source: Register, writer: anytype) !void {
        std.log.debug("cmp {s}, {s}", .{ @tagName(dest), @tagName(source) });

        _ = writer;

        std.log.debug("{x}", .{modrm(dest, source)});
    }

    fn cmp_register_with_const(reg: Register, value: [4]u8, writer: anytype) !void {
        std.log.debug("cmp {s}, {x}", .{ @tagName(reg), std.mem.bytesToValue(u32, &value) });

        switch (reg) {
            .Rax, .Rcx, .Rdx, .Rdi, .Rsi => try writer.writeByte(0x48),
            .R8, .R9, .R10, .R11 => try writer.writeByte(0x4d),
            .Rsp, .Rbp => unreachable,
        }

        if (reg != .Rax) {
            try writer.writeByte(0x81);
        }

        switch (reg) {
            .Rax => try writer.writeByte(0x3d),
            .Rcx => try writer.writeByte(0xf9),
            .Rdx => try writer.writeByte(0xfa),
            .Rdi => try writer.writeByte(0xff),
            .Rsi => try writer.writeByte(0xfe),
            .R8 => try writer.writeByte(0xf8),
            .R9 => try writer.writeByte(0xf9),
            .R10 => try writer.writeByte(0xfa),
            .R11 => try writer.writeByte(0xfb),
            .Rsp, .Rbp => unreachable,
        }

        try writer.writeAll(&value);
    }
};

pub fn generate_kernel(allocator: mem.Allocator, block: *const ir.Block) ![]const u8 {
    var code = std.ArrayList(u8).init(allocator);
    var writer = code.writer();
    var store = try ValueStore.init(allocator, &writer);
    var labels = std.AutoHashMap(ir.Step, usize).init(allocator);

    var ctx = Context{
        .block = block,
        .cursor = 0,
        .code = &code,
        .writer = &writer,
        .store = &store,
        .labels = &labels,
    };

    try generate_prologue(&ctx);

    while (ctx.cursor < block.nodes.items.len) {
        try generate_node(&ctx);
    }

    try generate_epilogue(&ctx);

    return code.toOwnedSlice();
}

fn generate_node(ctx: *Context) !void {
    const node = ctx.block.nodes.items[ctx.cursor];

    switch (node.op) {
        .DEFINE_GLOBAL => {},
        .DEFINE_ACC => try generate_define_acc(node, ctx),
        .CONST => try generate_const(node, ctx),
        .LOOP => try generate_loop(node, ctx),
        .LOAD => try generate_load(node, ctx),
        .ALU => {},
        .UPDATE => {},
        .ENDLOOP => try generate_endloop(node, ctx),
        .STORE => {},
    }

    ctx.cursor += 1;
}

fn generate_define_acc(node: ir.Node, ctx: *Context) !void {
    var value = [_]u8{0} ** 4;

    try value_to_4_bytes(node.dtype.?, &value, node.arg.DEFINE_ACC);

    try ctx.store.write_value_static(node.step, value);
}

fn value_to_4_bytes(dtype: ir.DataTypes, dest: *[4]u8, source: []const u8) !void {
    switch (dtype) {
        .Int => {
            const T = i32;
            const int = try fmt.parseInt(T, source, 10);
            const int_little = std.mem.nativeToLittle(T, int);
            const bytes = std.mem.toBytes(int_little)[0..4];
            @memcpy(dest, bytes);
        },
        .Float => {
            const T = f32;
            const float = try fmt.parseFloat(T, source);
            const float_little = std.mem.nativeToLittle(T, float);
            const bytes = std.mem.toBytes(float_little)[0..4];
            @memcpy(dest, bytes);
        },
        else => unreachable,
    }
}

fn generate_load(node: ir.Node, ctx: *Context) !void {
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const buffer_idx: u8 = @truncate(buffer.arg.DEFINE_GLOBAL.idx);

    // arg is [*][*]u8

    // buffer_addr: [*]u8 = ([*][*]u8)[idx]
    _ = try ctx.store.load_from_buffer(node.step, buffer_idx);

    // value : [0:4]u8 = buffer_addr[index:index+4]
    //const index_step = node.inputs.?[1];
    //const index_loc = try ctx.store.load_value(index_step);

    // rax: [*]u8 = arg[idx]
    // mov rax, qword ptr [rbp + idx]
    //try ctx.writer.writeAll(&[_]u8{ 0x48, 0x8b, 0x45 });
    //try ctx.writer.print("{x}", .{buffer_idx});

    // eax: u32 = rax[index:index+4]
    //try ctx.writer.writeAll(&[_]u8{ 0x67, 0x8b, 0x40 });
    //try write_value(index, ctx);
}

fn generate_const(node: ir.Node, ctx: *Context) !void {
    var value = [_]u8{0} ** 4;

    try value_to_4_bytes(node.dtype.?, &value, node.arg.CONST);

    try ctx.store.write_const(node.step, value);
}

fn generate_loop(node: ir.Node, ctx: *Context) !void {
    const start = node.inputs.?[0];

    // mov index, start
    try ctx.store.copy_from_static(node.step, start);

    // label
    std.log.debug("label_{x}:", .{ctx.code.items.len});
    try ctx.labels.put(node.step, ctx.code.items.len);
}

fn generate_endloop(node: ir.Node, ctx: *Context) !void {
    const index_step: ir.Step = node.inputs.?[0];
    // inc index
    try ctx.store.inc(index_step);
    // cmp index, end
    const end_step: ir.Step = ctx.block.nodes.items[node.inputs.?[0]].inputs.?[1];
    try ctx.store.cmp(index_step, end_step);
    // jle label
    const label = ctx.labels.get(index_step).?;
    const distance = ctx.code.items.len - label;
    assert(distance < 0xff);
    const offset: u8 = 0xfe - @as(u8, @truncate(distance));
    std.log.debug("jle {x}; label_{x}", .{ offset, label });
    try ctx.writer.writeAll(&[_]u8{ 0x7e, offset });
}

fn generate_prologue(ctx: *Context) !void {
    std.log.debug("push Rbp", .{});
    try ctx.writer.writeByte(0x55);
    std.log.debug("mov Rbp, Rdi", .{});
    try ctx.writer.writeAll(&[_]u8{ 0x48, 0x89, 0xfd });
}

fn generate_epilogue(ctx: *Context) !void {
    std.log.debug("pop Rbp", .{});
    try ctx.writer.writeByte(0x5d);
    std.log.debug("ret", .{});
    try ctx.writer.writeByte(0xc3);
}
