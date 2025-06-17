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
        .@"enum" = .{
            .tag_type = usize,
            .fields = fields[0..idx],
            .decls = &[_]std.builtin.Type.Declaration{}, // has to be empty
            .is_exhaustive = true,
        },
    });
};

fn rex_prefix(dest: Register, source: ?Register, reg: ?Register) ?u8 {
    var rex: u8 = 0b01000000;

    // 64 bit registers
    rex |= 0b00001000;

    if (is_extended(dest)) {
        rex |= 0b00000100;
    }

    if (source) |src| {
        if (is_extended(src)) {
            rex |= 0b00000001;
        }
    }

    if (reg) |offset| {
        if (is_extended(offset)) {
            rex |= 0b00000010;
        }
    }

    if (rex == 0b01000000) {
        return null;
    } else {
        return rex;
    }
}

fn is_extended(reg: Register) bool {
    return switch (reg) {
        .R8, .R9, .R10, .R11 => true,
        else => false,
    };
}

fn reg_encoding(reg: Register) u3 {
    return switch (reg) {
        .Rax => 0b000,
        .Rcx => 0b001,
        .Rdx => 0b010,
        .Rsp => 0b100,
        .Rbp => 0b101,
        .Rsi => 0b110,
        .Rdi => 0b111,

        .R8 => 0b000,
        .R9 => 0b001,
        .R10 => 0b010,
        .R11 => 0b011,
    };
}

const ModRMOptions = union(enum) {
    displacement: enum {
        None,
        Byte,
        Word,
    },
    register: void,

    pub fn value(self: ModRMOptions) u2 {
        return switch (self) {
            .displacement => |displacement| switch (displacement) {
                .None => 0b00,
                .Byte => 0b01,
                .Word => 0b10,
            },
            .register => 0b11,
        };
    }
};

fn modrm(dest: Register, source: Register, options: ModRMOptions) u8 {
    const mod: u8 = options.value();

    const reg: u8 = reg_encoding(dest);

    const rm: u8 = reg_encoding(source);

    return mod << 6 | reg << 3 | rm;
}

fn CircularBuffer(comptime T: type) type {
    if (@typeInfo(T) != .@"enum") @compileError("must be enum");

    const size = std.meta.fields(T).len;

    return struct {
        const Self = @This();

        buffer: [size]T,
        head: usize,

        pub fn init() Self {
            var self = Self{
                .buffer = undefined,
                .head = 0,
            };

            for (std.meta.tags(T), 0..) |tag, idx| {
                self.buffer[idx] = tag;
            }

            return self;
        }

        pub fn read(self: *Self) T {
            const elem = self.buffer[self.head];

            self.head = (self.head + 1) % size;

            return elem;
        }
    };
}

const ValueStore = struct {
    map: std.AutoHashMap(ir.Step, Location),
    static_list: std.ArrayList(StaticRegister),
    temp_queue: CircularBuffer(TempRegister),
    temp_steps: std.AutoHashMap(TempRegister, ir.Step),
    consts: std.AutoHashMap(ir.Step, []const u8),
    allocator: mem.Allocator,
    writer: *std.ArrayList(u8).Writer,
    offset: i32 = 0,

    pub fn init(allocator: mem.Allocator, writer: anytype) !ValueStore {
        var value_store = ValueStore{
            .map = std.AutoHashMap(ir.Step, Location).init(allocator),
            .static_list = std.ArrayList(StaticRegister).init(allocator),
            .temp_queue = CircularBuffer(TempRegister).init(),
            .temp_steps = std.AutoHashMap(TempRegister, ir.Step).init(allocator),
            .consts = std.AutoHashMap(ir.Step, []const u8).init(allocator),
            .allocator = allocator,
            .writer = writer,
        };

        try value_store.static_list.appendSlice(std.meta.tags(StaticRegister));

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
        if (self.static_list.pop()) |reg| {
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
        const reg = self.temp_queue.read();

        if (self.temp_steps.get(reg)) |prev| {
            _ = self.temp_steps.remove(reg);
            _ = self.map.remove(prev);
        }

        const loc = Location{ .Temp = reg };

        try write_value_to_reg(reg.to_generic(), value, self.writer);

        try self.map.put(step, loc);
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
    pub fn load_from_buffer(self: *ValueStore, step: ir.Step, buffer_idx: usize, index: ir.Step) !void {
        const reg = self.temp_queue.read();

        const offset: usize = buffer_idx * 8;
        const bytes = std.mem.toBytes(offset)[0..4];

        try load_value_from_register(reg.to_generic(), .Rbp, bytes.*, self.writer);

        if (self.map.get(index)) |loc| {
            const index_reg = switch (loc) {
                .Temp => |index_reg| index_reg.to_generic(),
                .Static => |index_reg| index_reg.to_generic(),
                .Stack => |_| {
                    std.debug.panic("load_from_buffer: stack offset not implemented", .{});
                },
            };

            try load_value_from_register_register_offset(reg.to_generic(), reg.to_generic(), index_reg, self.writer);
        }

        const loc = Location{ .Temp = reg };

        try self.map.put(step, loc);

        try self.temp_steps.put(reg, step);
    }

    fn load_value_from_register(dest: Register, source: Register, idx: [4]u8, writer: anytype) !void {
        std.log.debug("mov {s}, qword ptr [{s} + 0x{x}]", .{ @tagName(dest), @tagName(source), std.mem.bytesToValue(u32, &idx) });

        if (rex_prefix(dest, source, null)) |prefix| {
            try writer.writeByte(prefix);
        }

        // op code
        try writer.writeByte(0x8b);

        try writer.writeByte(modrm(dest, source, .{ .displacement = .Word }));

        // SIB byte for [RSP] with no index
        // TODO: refuse to believe this is correct
        if (source == .Rsp) {
            try writer.writeByte(0x24);
        }

        try writer.writeAll(&idx);
    }

    fn load_value_from_register_register_offset(dest: Register, source: Register, offset: Register, writer: anytype) !void {
        std.log.debug("mov {s}, qword ptr [{s} + {s} * 8]", .{ @tagName(dest), @tagName(source), @tagName(offset) });

        if (rex_prefix(dest, source, null)) |prefix| {
            try writer.writeByte(prefix);
        }

        // op code
        try writer.writeByte(0x8b);

        // ModR/M byte: Mod = 00 (no displacement), Reg = dest, R/M = 100 (SIB follows)
        const modrm_sib: u8 = (@as(u8, @intCast(reg_encoding(dest))) << 3) | 0b100;
        try writer.writeByte(modrm_sib);

        // SIB byte: Scale = 0b00 (multiply index by 0), Index = offset, Base = source
        const sib: u8 = (0b11 << 6) | (@as(u8, @intCast(reg_encoding(offset))) << 3) | reg_encoding(source);
        try writer.writeByte(sib);
    }

    pub fn write_const(self: *ValueStore, step: ir.Step, value: [4]u8) !void {
        const buffer = try self.allocator.dupe(u8, &value);
        try self.consts.put(step, buffer);
    }

    pub fn copy_from_static(self: *ValueStore, dest: ir.Step, source: ir.Step) !void {
        if (self.map.get(source)) |source_loc| {
            if (self.static_list.pop()) |dest_reg| {
                const loc = Location{ .Static = dest_reg };
                try self.map.put(dest, loc);

                switch (source_loc) {
                    .Temp => |source_reg| {
                        const empty = [_]u8{0} ** 4;
                        try load_value_from_register(dest_reg.to_generic(), source_reg.to_generic(), empty, self.writer);
                    },
                    .Static => |source_reg| {
                        const empty = [_]u8{0} ** 4;
                        try load_value_from_register(dest_reg.to_generic(), source_reg.to_generic(), empty, self.writer);
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
                if (self.static_list.pop()) |dest_reg| {
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

        const modrm_inc: u8 = @as(u8, 0b11_000_000) | reg_encoding(reg);
        try writer.writeByte(modrm_inc);
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

        std.log.debug("{x}", .{modrm(dest, source, .{ .displacement = .Byte })});
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

    pub fn alu(self: *ValueStore, step: ir.Step, op: std.meta.fieldInfo(ir.Ops.Arg, .ALU).type, lhs: ir.Step, rhs: ir.Step) !void {
        const lhs_reg = self.get_register(lhs) catch |err| {
            std.debug.panic("Could not get lhs register: {}", .{err});
        };

        const rhs_reg = self.get_register(rhs) catch |err| {
            std.debug.panic("Could not get rhs register: {}", .{err});
        };

        const result_reg = self.temp_queue.read();

        try copy(result_reg.to_generic(), lhs_reg, self.writer);

        try alu_op(op, result_reg.to_generic(), rhs_reg, self.writer);

        const loc = Location{ .Temp = result_reg };
        try self.map.put(step, loc);
        try self.temp_steps.put(result_reg, step);
    }

    fn alu_op(op: std.meta.fieldInfo(ir.Ops.Arg, .ALU).type, lhs: Register, rhs: Register, writer: anytype) !void {
        std.log.debug("{s} {s}, {s}", .{ @tagName(op), @tagName(lhs), @tagName(rhs) });

        if (rex_prefix(rhs, lhs, null)) |prefix| {
            try writer.writeByte(prefix);
        }

        // use imul and idiv
        // care for clobbering, will prolly need to refactor some stuff
        // TODO: clobbering
        const opcode: []const u8 = switch (op) {
            .Add => ([_]u8{0x01})[0..], // ADD
            .Mul => ([_]u8{ 0x0F, 0xaf })[0..], // IMUL
            else => {
                std.debug.panic("Unsupported ALU operation", .{});
            },
        };
        try writer.writeAll(opcode);

        const modrm_byte = switch (op) {
            .Add => modrm(rhs, lhs, .{ .register = {} }),
            .Mul => modrm(lhs, rhs, .{ .register = {} }), // imul swaps, idk why but its annoying
            else => {
                std.debug.panic("Unsupported ALU operation", .{});
            },
        };
        try writer.writeByte(modrm_byte);
    }

    fn copy(dest: Register, source: Register, writer: anytype) !void {
        std.log.debug("mov {s}, {s}", .{ @tagName(dest), @tagName(source) });

        if (rex_prefix(source, dest, null)) |prefix| {
            try writer.writeByte(prefix);
        }

        try writer.writeByte(0x89);

        // Mod = 11 (register-to-register)
        const modrm_byte = modrm(source, dest, .{ .register = {} });
        try writer.writeByte(modrm_byte);
    }

    fn get_register(self: *ValueStore, step: ir.Step) !Register {
        if (self.map.get(step)) |loc| {
            return switch (loc) {
                .Temp => |reg| reg.to_generic(),
                .Static => |reg| reg.to_generic(),
                .Stack => |_| {
                    std.debug.panic("get_register: unimplemented stack storage, should pop into temp reg", .{});
                },
            };
        } else if (self.consts.get(step)) |value| {
            try self.write_value_temp(step, value[0..4].*);
            return self.get_register(step);
        } else {
            return error.DeallocatedTempRegister;
        }
    }

    pub fn store(self: *ValueStore, buffer_idx: usize, index: ir.Step, value: ir.Step) !void {
        const reg_base = self.temp_queue.read();
        const offset = buffer_idx * 8;
        const bytes = std.mem.toBytes(offset)[0..4];

        try load_value_from_register(reg_base.to_generic(), .Rbp, bytes.*, self.writer);

        const index_reg = self.get_register(index) catch |err| {
            std.debug.panic("Failed to get index register: {}", .{err});
        };

        const value_reg = self.get_register(value) catch |err| {
            std.debug.panic("Failed to get value register: {}", .{err});
        };

        try store_value_to_memory_with_offset(reg_base.to_generic(), index_reg, value_reg, self.writer);
    }

    fn store_value_to_memory_with_offset(dest: Register, index: Register, value: Register, writer: anytype) !void {
        std.log.debug("mov qword ptr [{s} + {s} * 8], {s}", .{ @tagName(dest), @tagName(index), @tagName(value) });

        if (rex_prefix(value, dest, index)) |prefix| {
            try writer.writeByte(prefix);
        }

        try writer.writeByte(0x89);

        // Write the ModR/M byte: Mod = 00 (memory), Reg = value, R/M = 100 (SIB follows)
        const modrm_byte = (0b00 << 6) | (@as(u8, @intCast(reg_encoding(value))) << 3) | 0b100;
        try writer.writeByte(modrm_byte);

        // Write the SIB byte: Scale = 0b00 (no scaling), Index = index, Base = base
        const sib_byte = (0b11 << 6) | (@as(u8, @intCast(reg_encoding(index))) << 3) | reg_encoding(dest);
        try writer.writeByte(sib_byte);
    }

    pub fn update(self: *ValueStore, variable: ir.Step, value: ir.Step) !void {
        const var_reg = self.get_register(variable) catch |err| {
            std.debug.panic("Failed to get var register: {}", .{err});
        };

        const value_reg = self.get_register(value) catch |err| {
            std.debug.panic("Failed to get value register: {}", .{err});
        };

        try copy(var_reg, value_reg, self.writer);
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
        .ALU => try generate_alu(node, ctx),
        .UPDATE => try generate_update(node, ctx),
        .ENDLOOP => try generate_endloop(node, ctx),
        .STORE => try generate_store(node, ctx),
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

fn generate_load(node: ir.Node, ctx: *Context) !void {
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const buffer_idx = buffer.arg.DEFINE_GLOBAL.idx;

    const index = node.inputs.?[1];

    try ctx.store.load_from_buffer(node.step, buffer_idx, index);
}

fn generate_alu(node: ir.Node, ctx: *Context) !void {
    try ctx.store.alu(node.step, node.arg.ALU, node.inputs.?[0], node.inputs.?[1]);
}

fn generate_update(node: ir.Node, ctx: *Context) !void {
    const variable = node.inputs.?[0];
    const value = node.inputs.?[1];

    try ctx.store.update(variable, value);
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
    std.log.debug("jl {x}; label_{x}", .{ offset, label });
    try ctx.writer.writeAll(&[_]u8{ 0x7c, offset });
}

fn generate_store(node: ir.Node, ctx: *Context) !void {
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const buffer_idx = buffer.arg.DEFINE_GLOBAL.idx;

    const index = node.inputs.?[1];
    const value = node.inputs.?[2];
    try ctx.store.store(buffer_idx, index, value);
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
