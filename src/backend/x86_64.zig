const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const fmt = std.fmt;
const io = std.io;
const log = std.log;
const assert = std.debug.assert;

const ir = @import("../compiler/ir.zig");

pub const Register = enum(u8) {
    // static
    Rax,
    Rcx,
    Rdx, // function arg 1
    Rdi,
    Rsi,

    // temp
    R8,
    R9,
    R10,
    R11,

    // special
    Rsp,
    Rbp,

    pub fn is_temp(self: @This()) bool {
        return switch (self) {
            .R8, .R9, .R10, .R11 => true,
            else => false,
        };
    }

    pub fn is_extended(self: @This()) bool {
        return switch (self) {
            .R8, .R9, .R10, .R11 => true,
            else => false,
        };
    }

    pub fn reg_encoding(self: @This()) u3 {
        return switch (self) {
            .Rax, .R8 => 0b000,
            .Rcx, .R9 => 0b001,
            .Rdx, .R10 => 0b010,
            .R11 => 0b011,
            .Rsp => 0b100,
            .Rbp => 0b101,
            .Rsi => 0b110,
            .Rdi => 0b111,
        };
    }
};

const CircularBuffer = struct {
    const Self = @This();

    const size = blk: {
        var _size = 0;
        for (meta.tags(Register)) |tag| if (tag.is_temp()) {
            _size += 1;
        };
        break :blk _size;
    };

    buffer: [size]Register,
    head: usize,

    pub fn init() Self {
        comptime var self = Self{ .buffer = undefined, .head = 0 };

        comptime var idx = 0;
        inline for (comptime meta.tags(Register)) |tag| comptime if (tag.is_temp()) {
            self.buffer[idx] = tag;
            idx += 1;
        };

        return self;
    }

    pub fn read(self: *Self) Register {
        const elem = self.buffer[self.head];

        self.head = (self.head + 1) % size;

        return elem;
    }
};

const Memory = struct {
    const Self = @This();

    map: std.AutoHashMap(ir.Step, Location),
    temp_queue: CircularBuffer,
    temp_steps: std.AutoHashMap(Register, ir.Step),
    consts: std.AutoHashMap(ir.Step, []const u8),
    allocator: mem.Allocator,
    offset: i32 = 0,
    encoder: *Encoder,

    pub fn init(allocator: mem.Allocator, encoder: *Encoder) !Self {
        return .{
            .map = std.AutoHashMap(ir.Step, Location).init(allocator),
            .temp_queue = CircularBuffer.init(),
            .temp_steps = std.AutoHashMap(Register, ir.Step).init(allocator),
            .consts = std.AutoHashMap(ir.Step, []const u8).init(allocator),
            .allocator = allocator,
            .encoder = encoder,
        };
    }

    pub const Location = union(enum) {
        Temp: Register,
        Stack: i32,
    };

    pub const SourceValue = union(enum) {
        Immediate: []const u8,
        Register: Register,
    };

    pub fn read(self: *Self, step: ir.Step) !Register {
        if (self.consts.get(step)) |value| {
            return try self.alloc_temp(step, value, self.encoder);
        }

        if (self.map.get(step)) |loc| {
            return switch (loc) {
                .Stack => |offset| blk: {
                    const reg = self.temp_queue.read();
                    try self.encoder.mov_reg_from_mem(reg, .Rbp, offset);
                    break :blk reg;
                },
                .Temp => |reg| reg,
            };
        }

        @panic("Failed to get register for step");
    }

    fn alloc_temp(self: *Self, step: ir.Step, value: []const u8, encoder: *Encoder) !Register {
        const reg = self.temp_queue.read();

        if (self.temp_steps.get(reg)) |previous_step_using_this_reg| {
            try self.write(previous_step_using_this_reg, .{ .Register = reg });
            _ = self.temp_steps.remove(reg);
            _ = self.map.remove(previous_step_using_this_reg);
        }

        const new_loc = Location{ .Temp = reg };
        try encoder.mov_reg_imm32(reg, value);
        try self.map.put(step, new_loc);
        try self.temp_steps.put(reg, step);

        return reg;
    }

    pub fn write(self: *Self, step: ir.Step, source: SourceValue) !void {
        const loc = if (self.map.get(step)) |loc|
            loc
        else {
            self.offset += 8;
            try self.map.put(step, .{ .Stack = -self.offset });

            switch (source) {
                .Immediate => |value| try self.encoder.push_imm32(value),
                .Register => |reg| try self.encoder.push_reg(reg),
            }

            return;
        };

        switch (loc) {
            .Stack => |offset| switch (source) {
                .Immediate => |value| {
                    try self.encoder.mov_mem_imm32(.Rbp, offset, value);
                },
                .Register => |reg| {
                    try self.encoder.mov_mem_from_reg(.Rbp, offset, reg);
                },
            },
            .Temp => |reg_dest| {
                switch (source) {
                    .Immediate => |value| {
                        try self.encoder.mov_reg_imm32(reg_dest, value);
                    },
                    .Register => |reg_src| {
                        try self.encoder.mov_reg_reg(reg_dest, reg_src);
                    },
                }
            },
        }
    }
};

const Encoder = struct {
    const Self = @This();

    buffer: std.ArrayList(u8),
    allocator: mem.Allocator,

    // Prefixes
    const REX_W = 0x48; // 64-bit operand size prefix

    // Control Flow
    const JL_SHORT = 0x7c;
    const RET = 0xc3;

    // Stack
    const PUSH_IMM32 = 0x68;
    const LEAVE = 0xc9;

    // Data Movement
    const MOV_REG_REG = 0x89;
    const MOV_REG_FROM_MEM = 0x8b;
    const MOV_REG_IMM = 0xc7;

    // Arithmetic & Logic
    const ADD_REG_REG = 0x01; // ADD r/m64, r64
    const IMUL_REG_REG = 0x0f_af; // IMUL r64, r/m64
    const XOR_REG_REG = 0x31; // XOR r/m64, r64
    const INC_REG = 0xff; // INC r/m64
    const CMP_REG_REG = 0x39; // CMP r/m64, r64
    const CMP_REG_IMM = 0x81; // CMP r/m64, imm32
    const CMP_RAX_IMM = 0x3d; // CMP RAX, imm32

    pub fn init(allocator: mem.Allocator) Self {
        return .{
            .buffer = std.ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn epilogue(self: *Self) !void {
        log.debug("leave", .{});
        try self.buffer.writer().writeByte(LEAVE);
        log.debug("ret", .{});
        try self.buffer.writer().writeByte(RET);
    }

    pub fn mov_reg_imm32(self: *Self, dest: Register, value: []const u8) !void {
        if (mem.bytesToValue(u32, value[0..4]) == 0) {
            log.debug("xor {s}, {s}", .{ @tagName(dest), @tagName(dest) });
            try self.emit_rex(.{ .reg = dest, .rm = dest });
            try self.buffer.writer().writeByte(XOR_REG_REG);
            try self.buffer.writer().writeByte(modrm(dest, dest, .{ .register = {} }));
            return;
        }

        log.debug("mov {s}, {x}", .{ @tagName(dest), value });
        try self.emit_rex(.{ .rm = dest });
        try self.buffer.writer().writeByte(MOV_REG_IMM);

        // ModR/M: Mod=11 (reg), R/M=dest, Reg=0 (opcode extension)
        const modrm_byte = (@as(u8, 0b11) << 6) | (0b000 << 3) | dest.reg_encoding();
        try self.buffer.writer().writeByte(modrm_byte);
        try self.buffer.writer().writeAll(value[0..4]);
    }

    pub fn mov_reg_reg(self: *Self, dest: Register, src: Register) !void {
        log.debug("mov {s}, {s}", .{ @tagName(dest), @tagName(src) });

        try self.emit_rex(.{ .reg = src, .rm = dest });
        try self.buffer.writer().writeByte(MOV_REG_REG);
        try self.buffer.writer().writeByte(modrm(src, dest, .{ .register = {} }));
    }

    pub fn push_imm32(self: *Self, value: []const u8) !void {
        log.debug("push {x}", .{value});
        try self.buffer.writer().writeByte(PUSH_IMM32);
        try self.buffer.writer().writeAll(value[0..4]);
    }

    pub fn mov_reg_from_mem(self: *Self, dest: Register, base: Register, disp: i32) !void {
        log.debug("mov {s}, qword ptr [{s} + 0x{x}]", .{ @tagName(dest), @tagName(base), disp });
        try self.emit_rex(.{ .reg = dest, .rm = base });
        try self.buffer.writer().writeByte(MOV_REG_FROM_MEM);

        // Assuming 32-bit displacement for simplicity
        try self.buffer.writer().writeByte(modrm(dest, base, .{ .displacement = .Word }));

        // SIB byte is needed if base register is RSP
        if (base == .Rsp) {
            try self.buffer.writer().writeByte(0x24);
        }

        try self.buffer.writer().writeInt(i32, disp, .little);
    }

    pub fn mov_reg_from_mem_sib(self: *Self, dest: Register, base: Register, index: Register, comptime scale: u2) !void {
        log.debug("mov {s}, qword ptr [{s} + {s} * {d}]", .{ @tagName(dest), @tagName(base), @tagName(index), 1 << scale });
        try self.emit_rex(.{ .reg = dest, .rm = base, .index = index });
        try self.buffer.writer().writeByte(MOV_REG_FROM_MEM);

        // ModR/M byte: Mod = 00 (no displacement), Reg = dest, R/M = 100 (SIB follows)
        const modrm_byte: u8 = (0b00 << 6) | (@as(u8, dest.reg_encoding()) << 3) | 0b100;
        try self.buffer.writer().writeByte(modrm_byte);

        // SIB byte: Scale, Index, Base
        const sib_byte: u8 = (@as(u8, scale) << 6) | (@as(u8, index.reg_encoding()) << 3) | base.reg_encoding();
        try self.buffer.writer().writeByte(sib_byte);
    }

    pub fn mov_mem_from_reg(self: *Self, base: Register, disp: i32, src: Register) !void {
        log.debug("mov qword ptr [{s} + 0x{x}], {s}", .{ @tagName(base), disp, @tagName(src) });
        try self.emit_rex(.{ .reg = src, .rm = base });
        try self.buffer.writer().writeByte(MOV_REG_REG);

        // ModR/M: Mod=10 (disp32), Reg=src, RM=base
        try self.buffer.writer().writeByte(modrm(src, base, .{ .displacement = .Word }));

        // SIB byte is needed if base register is RSP
        if (base == .Rsp) {
            try self.buffer.writer().writeByte(0x24);
        }
        try self.buffer.writer().writeInt(i32, disp, .little);
    }

    pub fn mov_mem_sib_from_reg(self: *Self, base: Register, index: Register, src: Register, comptime scale: u2) !void {
        log.debug("mov qword ptr [{s} + {s} * {d}], {s}", .{ @tagName(base), @tagName(index), 1 << scale, @tagName(src) });
        try self.emit_rex(.{ .reg = src, .rm = base, .index = index });
        try self.buffer.writer().writeByte(MOV_REG_REG); // Opcode is 0x89 for mem<-reg

        // ModR/M byte: Mod = 00 (no displacement), Reg = src, R/M = 100 (SIB follows)
        const modrm_byte = (0b00 << 6) | (@as(u8, src.reg_encoding()) << 3) | 0b100;
        try self.buffer.writer().writeByte(modrm_byte);

        // SIB byte: Scale, Index, Base
        const sib_byte = (@as(u8, scale) << 6) | (@as(u8, index.reg_encoding()) << 3) | base.reg_encoding();
        try self.buffer.writer().writeByte(sib_byte);
    }

    pub fn mov_mem_imm32(self: *Self, base: Register, disp: i32, value: []const u8) !void {
        log.debug("mov qword ptr [{s} + 0x{x}], {x}", .{ @tagName(base), disp, value });
        try self.emit_rex(.{ .rm = base });
        try self.buffer.writer().writeByte(MOV_REG_IMM);

        // ModR/M byte for [base + disp32] addressing mode.
        const modrm_byte = (0b10 << 6) | (@as(u8, 0b000) << 3) | base.reg_encoding();
        try self.buffer.writer().writeByte(modrm_byte);
        if (base == .Rsp) {
            try self.buffer.writer().writeByte(0x24);
        }
        try self.buffer.writer().writeInt(i32, disp, .little);
        try self.buffer.writer().writeAll(value[0..4]);
    }

    pub fn inc_reg(self: *Self, reg: Register) !void {
        log.debug("inc {s}", .{@tagName(reg)});
        try self.emit_rex(.{ .rm = reg });
        try self.buffer.writer().writeByte(INC_REG);

        const modrm_inc: u8 = @as(u8, 0b11_000_000) | reg.reg_encoding();
        try self.buffer.writer().writeByte(modrm_inc);
    }

    pub fn cmp_reg_imm32(self: *Self, reg: Register, value: u32) !void {
        log.debug("cmp {s}, {x}", .{ @tagName(reg), value });
        try self.emit_rex(.{ .rm = reg });

        // Special shorter encoding for comparing with RAX
        if (reg == .Rax) {
            try self.buffer.writer().writeByte(CMP_RAX_IMM);
        } else {
            try self.buffer.writer().writeByte(CMP_REG_IMM);
            // ModR/M: Mod=11 (reg), R/M=reg, Reg=7 (opcode extension)
            const modrm_byte = (@as(u8, 0b11) << 6) | (0b111 << 3) | reg.reg_encoding();
            try self.buffer.writer().writeByte(modrm_byte);
        }

        try self.buffer.writer().writeInt(u32, value, .little);
    }

    pub fn cmp_reg_reg(self: *Self, reg1: Register, reg2: Register) !void {
        log.debug("cmp {s}, {s}", .{ @tagName(reg1), @tagName(reg2) });
        try self.emit_rex(.{ .reg = reg2, .rm = reg1 });
        try self.buffer.writer().writeByte(CMP_REG_REG);
        try self.buffer.writer().writeByte(modrm(reg2, reg1, .{ .register = {} }));
    }

    pub fn add_reg_reg(self: *Self, dest: Register, src: Register) !void {
        log.debug("add {s}, {s}", .{ @tagName(dest), @tagName(src) });
        try self.emit_rex(.{ .reg = dest, .rm = src });
        try self.buffer.writer().writeByte(ADD_REG_REG);
        try self.buffer.writer().writeByte(modrm(src, dest, .{ .register = {} }));
    }

    pub fn imul_reg_reg(self: *Self, dest: Register, src: Register) !void {
        log.debug("imul {s}, {s}", .{ @tagName(dest), @tagName(src) });
        try self.emit_rex(.{ .reg = dest, .rm = src });
        try self.buffer.writer().writeInt(u16, IMUL_REG_REG, .big);
        try self.buffer.writer().writeByte(modrm(dest, src, .{ .register = {} }));
    }

    pub fn jle(self: *Self, distance: usize) !void {
        assert(distance < 0xff);
        const offset: u8 = 0xfe - @as(u8, @truncate(distance));
        log.debug("jl 0x{x}", .{self.buffer.items.len - distance});
        try self.buffer.writer().writeAll(&[_]u8{ JL_SHORT, offset });
    }

    pub fn push_reg(self: *Self, reg: Register) !void {
        log.debug("push {s}", .{@tagName(reg)});

        if (reg.is_extended()) {
            try self.buffer.writer().writeByte(0x41);
        }
        try self.buffer.writer().writeByte(0x50 + @as(u8, reg.reg_encoding()));
    }

    const RexOperands = struct {
        reg: ?Register = null,
        rm: ?Register = null,
        index: ?Register = null,
        is_64bit: bool = true,
    };

    fn emit_rex(self: *Self, ops: RexOperands) !void {
        var prefix: u8 = 0b01000000;
        var needs_emit: bool = false;

        if (ops.is_64bit) {
            prefix |= 0b1000;
            needs_emit = true;
        }
        if (ops.reg) |r| {
            if (r.is_extended()) {
                prefix |= 0b0100; // REX.R
                needs_emit = true;
            }
        }
        if (ops.rm) |r| {
            if (r.is_extended()) {
                prefix |= 0b0001; // REX.B
                needs_emit = true;
            }
        }
        if (ops.index) |r| {
            if (r.is_extended()) {
                prefix |= 0b0010; // REX.X
                needs_emit = true;
            }
        }

        if (needs_emit) {
            try self.buffer.writer().writeByte(prefix);
        }
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

        const reg: u8 = dest.reg_encoding();

        const rm: u8 = source.reg_encoding();

        return mod << 6 | reg << 3 | rm;
    }
};

const Context = struct {
    block: *const ir.Block,
    store: *Memory,
    labels: *std.AutoHashMap(ir.Step, usize),
    encoder: *Encoder,
};

pub fn generate_kernel(allocator: mem.Allocator, block: *const ir.Block) ![]const u8 {
    var labels = std.AutoHashMap(ir.Step, usize).init(allocator);
    var encoder = Encoder.init(allocator);
    var store = try Memory.init(allocator, &encoder);

    var ctx = Context{
        .block = block,
        .store = &store,
        .labels = &labels,
        .encoder = &encoder,
    };

    try generate_prologue(&ctx);

    for (block.nodes.items) |node| switch (node.op) {
        .DEFINE_GLOBAL => {},
        .DEFINE_ACC => try generate_define_acc(node, &ctx),
        .CONST => try generate_const(node, &ctx),
        .LOOP => try generate_loop(node, &ctx),
        .LOAD => try generate_load(node, &ctx),
        .ALU => try generate_alu(node, &ctx),
        .UPDATE => try generate_update(node, &ctx),
        .ENDLOOP => try generate_endloop(node, &ctx),
        .STORE => try generate_store(node, &ctx),
    };

    try generate_epilogue(&ctx);

    return encoder.buffer.toOwnedSlice();
}

fn generate_define_acc(node: ir.Node, ctx: *Context) !void {
    try ctx.store.write(node.step, .{ .Immediate = node.arg.DEFINE_ACC });
}

fn generate_const(node: ir.Node, ctx: *Context) !void {
    try ctx.store.consts.put(node.step, node.arg.CONST);
}

fn generate_loop(node: ir.Node, ctx: *Context) !void {
    const step_start = node.inputs.?[0];

    if (ctx.store.consts.get(step_start)) |value| {
        try ctx.store.write(node.step, .{ .Immediate = value });
    }

    log.debug("label_0x{x}:", .{ctx.encoder.buffer.items.len});
    try ctx.labels.put(node.step, ctx.encoder.buffer.items.len);
}

fn generate_load(node: ir.Node, ctx: *Context) !void {
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const buffer_idx = buffer.arg.DEFINE_GLOBAL.idx;

    const index = node.inputs.?[1];

    const reg = ctx.store.temp_queue.read();
    try ctx.encoder.mov_reg_from_mem(reg, .Rdi, @intCast(buffer_idx * 8));

    const index_reg = try ctx.store.read(index);
    try ctx.encoder.mov_reg_from_mem_sib(reg, reg, index_reg, 0b11);

    try ctx.store.map.put(node.step, .{ .Temp = reg });
    try ctx.store.temp_steps.put(reg, node.step);
}

fn generate_alu(node: ir.Node, ctx: *Context) !void {
    const lhs_reg = try ctx.store.read(node.inputs.?[0]);
    const rhs_reg = try ctx.store.read(node.inputs.?[1]);

    const reg_result = ctx.store.temp_queue.read();

    try ctx.encoder.mov_reg_reg(reg_result, lhs_reg);

    switch (node.arg.ALU) {
        .Add => try ctx.encoder.add_reg_reg(reg_result, rhs_reg),
        .Mul => try ctx.encoder.imul_reg_reg(reg_result, rhs_reg),
        else => @panic("Unsupported ALU operation"),
    }

    try ctx.store.map.put(node.step, .{ .Temp = reg_result });
    try ctx.store.temp_steps.put(reg_result, node.step);
}

fn generate_update(node: ir.Node, ctx: *Context) !void {
    const step_variable = node.inputs.?[0];
    const step_value = node.inputs.?[1];

    const reg_value = try ctx.store.read(step_value);

    try ctx.store.write(step_variable, .{ .Register = reg_value });
}

fn generate_endloop(node: ir.Node, ctx: *Context) !void {
    const step_index: ir.Step = node.inputs.?[0];
    const loop_node = ctx.block.nodes.items[step_index];
    const step_end: ir.Step = loop_node.inputs.?[1];

    // inc
    const reg_index = try ctx.store.read(step_index);
    try ctx.encoder.inc_reg(reg_index);
    try ctx.store.write(step_index, .{ .Register = reg_index });

    // cmp
    const reg_end = try ctx.store.read(step_end);
    try ctx.encoder.cmp_reg_reg(reg_index, reg_end);

    // jmp
    const label = ctx.labels.get(step_index).?;
    const distance = ctx.encoder.buffer.items.len - label;
    try ctx.encoder.jle(distance);
}

fn generate_store(node: ir.Node, ctx: *Context) !void {
    // load addr for storage buffer
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const buffer_idx = buffer.arg.DEFINE_GLOBAL.idx;
    const reg_base = ctx.store.temp_queue.read();
    try ctx.encoder.mov_reg_from_mem(reg_base, .Rdi, @intCast(buffer_idx * 8));

    // load value
    const step_value = node.inputs.?[2];
    const reg_value = try ctx.store.read(step_value);

    // load index of buffer
    const step_index = node.inputs.?[1];
    const reg_index = try ctx.store.read(step_index);

    // store value into addr + index
    try ctx.encoder.mov_mem_sib_from_reg(reg_base, reg_index, reg_value, 0b11);
}

fn generate_prologue(ctx: *Context) !void {
    try ctx.encoder.push_reg(.Rbp);
    try ctx.encoder.mov_reg_reg(.Rbp, .Rsp);
}

fn generate_epilogue(ctx: *Context) !void {
    try ctx.encoder.epilogue();
}
