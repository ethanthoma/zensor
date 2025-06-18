const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const fmt = std.fmt;
const io = std.io;
const assert = std.debug.assert;

const ir = @import("../compiler/ir.zig");

const Context = struct {
    block: *const ir.Block,
    cursor: usize = 0,
    code: *std.ArrayList(u8),
    writer: *std.ArrayList(u8).Writer,
    store: *ValueStore,
    labels: *std.AutoHashMap(ir.Step, usize),
    encoder: *Encoder,
};

// rbp: [*][*]u8
// rdi: function arg 1
// rsi: function arg 2
pub const Register = enum(u8) {
    // static
    Rax,
    Rcx,
    Rdx,
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

    pub fn is_static(self: @This()) bool {
        return switch (self) {
            .Rax, .Rcx, .Rdx, .Rdi, .Rsi => true,
            else => false,
        };
    }

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

    const size = 12;

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

const ValueStore = struct {
    const Self = @This();

    map: std.AutoHashMap(ir.Step, Location),
    static_list: std.ArrayList(Register),
    temp_queue: CircularBuffer,
    temp_steps: std.AutoHashMap(Register, ir.Step),
    consts: std.AutoHashMap(ir.Step, []const u8),
    allocator: mem.Allocator,
    writer: *std.ArrayList(u8).Writer,
    offset: i32 = 0,

    pub fn init(allocator: mem.Allocator, writer: anytype) !Self {
        var value_store = Self{
            .map = std.AutoHashMap(ir.Step, Location).init(allocator),
            .static_list = std.ArrayList(Register).init(allocator),
            .temp_queue = CircularBuffer.init(),
            .temp_steps = std.AutoHashMap(Register, ir.Step).init(allocator),
            .consts = std.AutoHashMap(ir.Step, []const u8).init(allocator),
            .allocator = allocator,
            .writer = writer,
        };

        inline for (meta.tags(Register)) |tag| if (tag.is_static()) {
            try value_store.static_list.append(tag);
        };

        return value_store;
    }

    pub const Location = union(enum) {
        Temp: Register,
        Static: Register,
        Stack: i32,
    };

    pub const SourceValue = union(enum) {
        Immediate: u32,
        Variable: ir.Step,
    };

    pub fn load_from_buffer(self: *Self, step: ir.Step, buffer_idx: u32, index: ir.Step, encoder: *Encoder) !void {
        const reg = self.temp_queue.read();
        try encoder.mov_reg_from_mem(reg, .Rbp, buffer_idx * 8);

        const index_reg = try (self.peek(index) orelse error.InvalidIndex);
        try encoder.mov_reg_from_mem_sib(reg, reg, index_reg, 0b11);

        const loc = Location{ .Temp = reg };
        try self.map.put(step, loc);
        try self.temp_steps.put(reg, step);
    }

    pub fn write_const(self: *Self, step: ir.Step, value: []u8) !void {
        try self.consts.put(step, value);
    }

    pub fn assign_static(self: *Self, step_dest: ir.Step, source: SourceValue, encoder: *Encoder) !void {
        const loc_dest = try self.alloc_static_location(step_dest);

        switch (loc_dest) {
            .Static => |reg| switch (source) {
                .Immediate => |value| try encoder.mov_reg_imm32(reg, value),
                .Variable => |step_source| if (self.consts.get(step_source)) |bytes| {
                    try encoder.mov_reg_imm32(reg, mem.bytesToValue(u32, bytes));
                } else {
                    try encoder.push_reg(try self.ensure_in_register(step_source, encoder));
                },
            },
            .Stack => |_| switch (source) {
                .Immediate => |value| try encoder.push_imm32(value),
                .Variable => |step_source| if (self.consts.get(step_source)) |bytes| {
                    try encoder.push_imm32(mem.bytesToValue(u32, bytes));
                } else {
                    try encoder.push_reg(try self.ensure_in_register(step_source, encoder));
                },
            },
            .Temp => unreachable,
        }
    }

    pub fn cmp(self: *Self, dest: ir.Step, source: ir.Step, encoder: *Encoder) !void {
        const reg_dest = self.peek(dest) orelse {
            // only support comparing const and reg or reg and reg
            assert(self.map.contains(source));
            return self.cmp(source, dest, encoder);
        };

        if (self.peek(source)) |reg_src| {
            try encoder.cmp_reg_reg(reg_dest, reg_src);
        } else if (self.consts.get(source)) |const_value| {
            try encoder.cmp_reg_imm32(reg_dest, mem.bytesToValue(u32, const_value));
        }
    }

    pub fn alu(self: *Self, step: ir.Step, op: meta.fieldInfo(ir.Ops.Arg, .ALU).type, lhs: ir.Step, rhs: ir.Step, encoder: *Encoder) !void {
        const lhs_reg = try self.ensure_in_register(lhs, encoder);
        const rhs_reg = try self.ensure_in_register(rhs, encoder);

        const reg_result = self.temp_queue.read();

        try encoder.mov_reg_reg(reg_result, lhs_reg);

        switch (op) {
            .Add => try encoder.add_reg_reg(reg_result, rhs_reg),
            .Mul => try encoder.imul_reg_reg(reg_result, rhs_reg),
            else => @panic("Unsupported ALU operation"),
        }

        const loc = Location{ .Temp = reg_result };
        try self.map.put(step, loc);
        try self.temp_steps.put(reg_result, step);
    }

    fn peek(self: *const Self, step: ir.Step) ?Register {
        return if (self.map.get(step)) |loc| switch (loc) {
            .Temp => |reg| reg,
            .Static => |reg| reg,
            .Stack => |_| null,
        } else null;
    }

    fn ensure_in_register(self: *Self, step: ir.Step, encoder: *Encoder) !Register {
        return self.peek(step) orelse if (self.consts.get(step)) |value|
            try self.alloc_temp(step, mem.bytesToValue(u32, value), encoder)
        else
            @panic("Failed to get register");
    }

    fn alloc_temp(self: *Self, step: ir.Step, value: u32, encoder: *Encoder) !Register {
        const reg = self.temp_queue.read();

        if (self.temp_steps.get(reg)) |previous_step_using_this_reg| {
            _ = self.temp_steps.remove(reg);
            _ = self.map.remove(previous_step_using_this_reg);
        }

        const new_loc = Location{ .Temp = reg };
        try encoder.mov_reg_imm32(reg, value);
        try self.map.put(step, new_loc);
        try self.temp_steps.put(reg, step);

        return reg;
    }

    fn alloc_static_location(self: *Self, step: ir.Step) !Location {
        if (self.static_list.pop()) |reg| {
            const loc = Location{ .Static = reg };
            try self.map.put(step, loc);
            return loc;
        } else {
            self.offset += 4;
            const loc = Location{ .Stack = -self.offset };
            try self.map.put(step, loc);
            return loc;
        }
    }

    pub fn store(self: *Self, buffer_idx: u32, step_index: ir.Step, step_value: ir.Step, encoder: *Encoder) !void {
        const reg_base = self.temp_queue.read();

        try encoder.mov_reg_from_mem(reg_base, .Rbp, buffer_idx * 8);

        const reg_index = self.peek(step_index) orelse
            if (self.consts.get(step_index)) |value|
                try self.alloc_temp(step_index, mem.bytesToValue(u32, value), encoder)
            else
                @panic("Failed to get index register");

        const reg_value = self.peek(step_value) orelse
            if (self.consts.get(step_value)) |value|
                try self.alloc_temp(step_value, mem.bytesToValue(u32, value), encoder)
            else
                @panic("Failed to get value register");

        try encoder.mov_mem_sib_from_reg(reg_base, reg_index, reg_value, 0b11);
    }

    pub fn update(self: *Self, step_variable: ir.Step, step_value: ir.Step, encoder: *Encoder) !void {
        const reg_variable = self.peek(step_variable) orelse @panic("Variable must be a non const register");

        if (self.peek(step_value)) |reg_value| {
            try encoder.mov_reg_reg(reg_variable, reg_value);
        } else if (self.consts.get(step_value)) |value| {
            try encoder.mov_reg_imm32(reg_variable, mem.bytesToValue(u32, value));
        }
    }
};

const Encoder = struct {
    const Self = @This();

    writer: *std.ArrayList(u8).Writer,

    // Prefixes
    const REX_W = 0x48; // 64-bit operand size prefix

    // Control Flow
    const JLE_SHORT = 0x7e;
    const JL_SHORT = 0x7c;
    const RET = 0xc3;

    // Stack
    const PUSH_RBP = 0x55;
    const POP_RBP = 0x5d;
    const PUSH_IMM32 = 0x68;

    // Data Movement
    const MOV_REG_REG = 0x89; // Move register to register/memory
    const MOV_REG_FROM_MEM = 0x8b; // Move memory to register
    const MOV_REG_IMM = 0xc7; // Move immediate to register

    // Arithmetic & Logic
    const ADD_REG_REG = 0x01; // ADD r/m64, r64
    const IMUL_REG_REG = 0x0f_af; // IMUL r64, r/m64
    const XOR_REG_REG = 0x31; // XOR r/m64, r64
    const INC_REG = 0xff; // INC r/m64
    const CMP_REG_REG = 0x39; // CMP r/m64, r64
    const CMP_REG_IMM = 0x81; // CMP r/m64, imm32
    const CMP_RAX_IMM = 0x3d; // CMP RAX, imm32

    pub fn init(writer: *std.ArrayList(u8).Writer) Self {
        return Self{ .writer = writer };
    }

    pub fn prologue(self: *Self) !void {
        std.log.debug("push rbp", .{});
        try self.writer.writeByte(PUSH_RBP);
        std.log.debug("mov rbp, rdi", .{});
        // 0xfd is modrm for mov rbp, rdi
        try self.writer.writeAll(&[_]u8{ REX_W, MOV_REG_REG, 0xfd });
    }

    pub fn epilogue(self: *Self) !void {
        std.log.debug("pop rbp", .{});
        try self.writer.writeByte(POP_RBP);
        std.log.debug("ret", .{});
        try self.writer.writeByte(RET);
    }

    pub fn mov_reg_imm32(self: *Self, dest: Register, value: u32) !void {
        if (value == 0) {
            std.log.debug("xor {s}, {s}", .{ @tagName(dest), @tagName(dest) });
            try self.emit_rex(.{ .reg = dest, .rm = dest });
            try self.writer.writeByte(XOR_REG_REG);
            try self.writer.writeByte(modrm(dest, dest, .{ .register = {} }));
            return;
        }

        std.log.debug("mov {s}, {x}", .{ @tagName(dest), value });
        try self.emit_rex(.{ .rm = dest });
        try self.writer.writeByte(MOV_REG_IMM);

        // ModR/M: Mod=11 (reg), R/M=dest, Reg=0 (opcode extension)
        const modrm_byte = (@as(u8, 0b11) << 6) | (0b000 << 3) | dest.reg_encoding();
        try self.writer.writeByte(modrm_byte);
        try self.writer.writeInt(u32, value, std.builtin.Endian.little);
    }

    pub fn mov_reg_reg(self: *Self, dest: Register, src: Register) !void {
        std.log.debug("mov {s}, {s}", .{ @tagName(dest), @tagName(src) });

        try self.emit_rex(.{ .reg = src, .rm = dest });
        try self.writer.writeByte(MOV_REG_REG);
        try self.writer.writeByte(modrm(src, dest, .{ .register = {} }));
    }

    pub fn push_imm32(self: *Self, value: u32) !void {
        std.log.debug("push {x}", .{value});
        try self.writer.writeByte(PUSH_IMM32);
        try self.writer.writeInt(u32, value, .little);
    }

    pub fn mov_reg_from_mem(self: *Self, dest: Register, base: Register, disp: u32) !void {
        std.log.debug("mov {s}, qword ptr [{s} + 0x{x}]", .{ @tagName(dest), @tagName(base), disp });
        try self.emit_rex(.{ .reg = dest, .rm = base });
        try self.writer.writeByte(MOV_REG_FROM_MEM);

        // Assuming 32-bit displacement for simplicity
        try self.writer.writeByte(modrm(dest, base, .{ .displacement = .Word }));

        // SIB byte is needed if base register is RSP
        if (base == .Rsp) {
            try self.writer.writeByte(0x24);
        }

        try self.writer.writeInt(u32, disp, std.builtin.Endian.little);
    }

    pub fn mov_reg_from_mem_sib(self: *Self, dest: Register, base: Register, index: Register, comptime scale: u2) !void {
        std.log.debug("mov {s}, qword ptr [{s} + {s} * {d}]", .{ @tagName(dest), @tagName(base), @tagName(index), 1 << scale });
        try self.emit_rex(.{ .reg = dest, .rm = base, .index = index });
        try self.writer.writeByte(MOV_REG_FROM_MEM);

        // ModR/M byte: Mod = 00 (no displacement), Reg = dest, R/M = 100 (SIB follows)
        const modrm_byte: u8 = (0b00 << 6) | (@as(u8, dest.reg_encoding()) << 3) | 0b100;
        try self.writer.writeByte(modrm_byte);

        // SIB byte: Scale, Index, Base
        const sib_byte: u8 = (@as(u8, scale) << 6) | (@as(u8, index.reg_encoding()) << 3) | base.reg_encoding();
        try self.writer.writeByte(sib_byte);
    }

    pub fn mov_mem_sib_from_reg(self: *Self, base: Register, index: Register, src: Register, comptime scale: u2) !void {
        std.log.debug("mov qword ptr [{s} + {s} * {d}], {s}", .{ @tagName(base), @tagName(index), 1 << scale, @tagName(src) });
        try self.emit_rex(.{ .reg = src, .rm = base, .index = index });
        try self.writer.writeByte(MOV_REG_REG); // Opcode is 0x89 for mem<-reg

        // ModR/M byte: Mod = 00 (no displacement), Reg = src, R/M = 100 (SIB follows)
        const modrm_byte = (0b00 << 6) | (@as(u8, src.reg_encoding()) << 3) | 0b100;
        try self.writer.writeByte(modrm_byte);

        // SIB byte: Scale, Index, Base
        const sib_byte = (@as(u8, scale) << 6) | (@as(u8, index.reg_encoding()) << 3) | base.reg_encoding();
        try self.writer.writeByte(sib_byte);
    }

    pub fn inc_reg(self: *Self, reg: Register) !void {
        std.log.debug("inc {s}", .{@tagName(reg)});
        try self.emit_rex(.{ .rm = reg });
        try self.writer.writeByte(INC_REG);

        const modrm_inc: u8 = @as(u8, 0b11_000_000) | reg.reg_encoding();
        try self.writer.writeByte(modrm_inc);
    }

    pub fn cmp_reg_imm32(self: *Self, reg: Register, value: u32) !void {
        std.log.debug("cmp {s}, {x}", .{ @tagName(reg), value });
        try self.emit_rex(.{ .rm = reg });

        // Special shorter encoding for comparing with RAX
        if (reg == .Rax) {
            try self.writer.writeByte(CMP_RAX_IMM);
        } else {
            try self.writer.writeByte(CMP_REG_IMM);
            // ModR/M: Mod=11 (reg), R/M=reg, Reg=7 (opcode extension)
            const modrm_byte = (@as(u8, 0b11) << 6) | (0b111 << 3) | reg.reg_encoding();
            try self.writer.writeByte(modrm_byte);
        }

        try self.writer.writeInt(u32, value, .little);
    }

    pub fn cmp_reg_reg(self: *Self, reg1: Register, reg2: Register) !void {
        std.log.debug("cmp {s}, {s}", .{ @tagName(reg1), @tagName(reg2) });
        try self.emit_rex(.{ .reg = reg2, .rm = reg1 });
        try self.writer.writeByte(CMP_REG_REG);
        try self.writer.writeByte(modrm(reg2, reg1, .{ .register = {} }));
    }

    pub fn add_reg_reg(self: *Self, dest: Register, src: Register) !void {
        std.log.debug("add {s}, {s}", .{ @tagName(dest), @tagName(src) });
        try self.emit_rex(.{ .reg = dest, .rm = src });
        try self.writer.writeByte(ADD_REG_REG);
        try self.writer.writeByte(modrm(src, dest, .{ .register = {} }));
    }

    pub fn imul_reg_reg(self: *Self, dest: Register, src: Register) !void {
        std.log.debug("imul {s}, {s}", .{ @tagName(dest), @tagName(src) });
        try self.emit_rex(.{ .reg = dest, .rm = src });
        try self.writer.writeInt(u16, IMUL_REG_REG, .big);
        try self.writer.writeByte(modrm(dest, src, .{ .register = {} }));
    }

    pub fn jl_short(self: *Self, offset: i8) !void {
        std.log.debug("jl {d}", .{offset});
        try self.writer.writeByte(JL_SHORT);
        try self.writer.writeInt(i8, offset, .little);
    }

    pub fn push_reg(self: *Self, reg: Register) !void {
        std.log.debug("push {s}", .{@tagName(reg)});

        if (reg.is_extended()) {
            try self.writer.writeByte(0x41);
        }
        try self.writer.writeByte(0x50 + @as(u8, reg.reg_encoding()));
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
            try self.writer.writeByte(prefix);
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

pub fn generate_kernel(allocator: mem.Allocator, block: *const ir.Block) ![]const u8 {
    var code = std.ArrayList(u8).init(allocator);
    var writer = code.writer();
    var store = try ValueStore.init(allocator, &writer);
    var labels = std.AutoHashMap(ir.Step, usize).init(allocator);
    var encoder = Encoder.init(&writer);

    var ctx = Context{
        .block = block,
        .cursor = 0,
        .code = &code,
        .writer = &writer,
        .store = &store,
        .labels = &labels,
        .encoder = &encoder,
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
    const value = try fmt.parseInt(u32, node.arg.DEFINE_ACC, 10);

    try ctx.store.assign_static(node.step, .{ .Immediate = value }, ctx.encoder);
}

inline fn value_to_4_bytes(dtype: ir.DataTypes, dest: *[]u8, source: []const u8) !void {
    const bytes = switch (dtype) {
        .Int => mem.toBytes(try fmt.parseInt(i32, source, 10)),
        .Float => mem.toBytes(try fmt.parseFloat(f32, source)),
        else => unreachable,
    };

    @memcpy(dest.*, bytes[0..4]);
}

fn generate_const(node: ir.Node, ctx: *Context) !void {
    var value = try ctx.store.allocator.alloc(u8, 4);

    try value_to_4_bytes(node.dtype.?, &value, node.arg.CONST);

    try ctx.store.write_const(node.step, value);
}

fn generate_loop(node: ir.Node, ctx: *Context) !void {
    const start = node.inputs.?[0];

    // mov index, start
    try ctx.store.assign_static(node.step, .{ .Variable = start }, ctx.encoder);

    // label
    std.log.debug("label_{x}:", .{ctx.code.items.len});
    try ctx.labels.put(node.step, ctx.code.items.len);
}

fn generate_load(node: ir.Node, ctx: *Context) !void {
    const buffer = ctx.block.nodes.items[node.inputs.?[0]];
    const buffer_idx = buffer.arg.DEFINE_GLOBAL.idx;

    const index = node.inputs.?[1];

    try ctx.store.load_from_buffer(node.step, buffer_idx, index, ctx.encoder);
}

fn generate_alu(node: ir.Node, ctx: *Context) !void {
    try ctx.store.alu(node.step, node.arg.ALU, node.inputs.?[0], node.inputs.?[1], ctx.encoder);
}

fn generate_update(node: ir.Node, ctx: *Context) !void {
    const variable = node.inputs.?[0];
    const value = node.inputs.?[1];

    try ctx.store.update(variable, value, ctx.encoder);
}

fn generate_endloop(node: ir.Node, ctx: *Context) !void {
    const index_step: ir.Step = node.inputs.?[0];

    // inc
    try if (ctx.store.peek(index_step)) |reg| ctx.encoder.inc_reg(reg);

    const end_step: ir.Step = ctx.block.nodes.items[node.inputs.?[0]].inputs.?[1];
    try ctx.store.cmp(index_step, end_step, ctx.encoder);

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
    try ctx.store.store(buffer_idx, index, value, ctx.encoder);
}

fn generate_prologue(ctx: *Context) !void {
    try ctx.encoder.prologue();
}

fn generate_epilogue(ctx: *Context) !void {
    try ctx.encoder.epilogue();
}
