const std = @import("std");
const mem = std.mem;
const fmt = std.fmt;
const io = std.io;
const assert = std.debug.assert;

const ir = @import("../compiler/ir.zig");

// rdi: [*][*]u8

const Op = enum(u8) {
    ret = 0xc3,
    mov_r64_imm = 0xc7,
    xor = 0x31,
    inc = 0xff,
    cmp = 0x39,
    cmp_imm = 0x81,
};

const AOp = enum(u3) {
    cmp = 0b111,
};

const Register = enum(u4) {
    Rax = 0b0000,
    Rcx = 0b0001,
    Rdx = 0b0010,
    Rsi = 0b0110,
    Rdi = 0b0111,
    R8 = 0b1000,
    R9 = 0b1001,
    R10 = 0b1010,
    R11 = 0b1011,

    pub fn is_64_bit(reg: Register) bool {
        return switch (reg) {
            .Rax, .Rcx, .Rdx, .Rdi, .Rsi, .R8, .R9, .R10, .R11 => true,
        };
    }

    pub fn is_extended(reg: Register) bool {
        return @intFromEnum(reg) >= 8;
    }

    pub fn encoding(reg: Register) u3 {
        return @truncate(@intFromEnum(reg));
    }
};

const EAddr = struct {
    base: ?Register,
    index: ?Register = null,
    scale: u2 = 0,
    displacement: i32 = 0,

    pub fn addressing_mode(self: EAddr) Addressing_Mode {
        if (self.index != null) {
            return if (self.displacement == 0) .SIB else .SIBDisp32;
        } else if (self.base) |_| {
            if (self.displacement == 0) {
                return .Indirect;
            } else {
                return .IndirectDisp32;
            }
        } else {
            return .RIPRelative;
        }
    }
};

const Addressing_Mode = enum {
    Register,
    Indirect,
    IndirectDisp32,
    SIB,
    SIBDisp32,
    RIPRelative,
};

const Operand = union(enum) {
    Register: Register,
    Memory: EAddr,
    Immediate: []const u8,
};

const Instruction = struct {
    op: Op,
    dest: ?Register = null,
    source: ?Operand = null,

    pub fn calculate_rex(self: Instruction) ?u8 {
        var rex: u8 = 0x40;

        if (self.dest) |dest| {
            if (dest.is_64_bit()) {
                rex |= 0x48;
            }

            if (dest.is_extended()) {
                rex |= 0x44;
            }
        }

        if (self.source) |source| {
            switch (source) {
                .Memory => |addr| {
                    if (addr.index) |index| {
                        if (index.is_extended()) {
                            rex |= 0x42;
                        }
                    }

                    if (addr.base) |base| {
                        if (base.is_extended()) {
                            rex |= 0x41;
                        }
                    }
                },
                .Register => |reg| {
                    if (reg.is_64_bit()) {
                        rex |= 0x48;
                    }

                    if (reg.is_extended()) {
                        rex |= 0x41;
                    }
                },
                .Immediate => {},
            }
        }

        return if (rex != 0x40) rex else null;
    }

    pub fn calculate_modrm(self: Instruction) ?u8 {
        var mod: u2 = 0;
        var reg: u3 = 0;
        var rm: u3 = 0;

        if (self.dest) |dest| {
            reg = dest.encoding();
        }

        if (self.source) |source| {
            switch (source) {
                .Immediate => {},
                .Register => |register| {
                    rm = register.encoding();
                    mod = 0b11; // Register-to-register
                },
                .Memory => |addr| {
                    const addressing_mode = addr.addressing_mode();
                    switch (addressing_mode) {
                        .Register => {},
                        .Indirect => {
                            mod = 0b00;
                            rm = addr.base.?.encoding();
                        },
                        .IndirectDisp32 => {
                            mod = 0b10;
                            rm = addr.base.?.encoding();
                        },
                        .SIB, .SIBDisp32 => {
                            rm = 0b100; // Indicates SIB byte follows
                            mod = if (addressing_mode == .SIB) 0b00 else 0b10;
                        },
                        .RIPRelative => {
                            mod = 0b00;
                            rm = 0b101;
                        },
                    }
                },
            }
        }

        const modrm = (@as(u8, mod) << 6) | (@as(u8, reg) << 3) | rm;

        return if (modrm != 0) modrm else null;
    }

    fn encode(self: @This(), writer: anytype) !void {
        if (self.calculate_rex()) |rex| {
            try writer.writeByte(rex);
        }

        try writer.writeByte(@intFromEnum(self.op));

        if (self.calculate_modrm()) |modrm| {
            try writer.writeByte(modrm);
        }

        if (self.source) |source| {
            switch (source) {
                .Memory => |addr| {
                    const addressing_mode = addr.addressing_mode();
                    if (addressing_mode == .IndirectDisp32 or addressing_mode == .SIBDisp32 or addressing_mode == .RIPRelative) {
                        try writer.writeAll(&std.mem.toBytes(addr.displacement));
                    }
                },
                .Immediate => |immediate| {
                    try writer.writeAll(immediate);
                },
                .Register => {},
            }
        }
    }
};

fn CircularBuffer(comptime T: type) type {
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

const Context = struct {
    block: *const ir.Block,
    code: std.ArrayList(u8),
    labels: std.AutoHashMap(ir.Step, usize),
    consts: std.AutoHashMap(ir.Step, []const u8),
    registers: CircularBuffer(Register),
    map: std.AutoHashMap(ir.Step, Register),

    fn init(allocator: mem.Allocator, block: *const ir.Block) @This() {
        return .{
            .block = block,
            .code = std.ArrayList(u8).init(allocator),
            .labels = std.AutoHashMap(ir.Step, usize).init(allocator),
            .consts = std.AutoHashMap(ir.Step, []const u8).init(allocator),
            .registers = CircularBuffer(Register).init(),
            .map = std.AutoHashMap(ir.Step, Register).init(allocator),
        };
    }

    fn deinit(self: *@This()) void {
        self.code.deinit();
        self.labels.deinit();
        self.consts.deinit();
        self.map.deinit();
    }
};

pub fn generate_kernel(allocator: mem.Allocator, block: *const ir.Block) ![]const u8 {
    var ctx = Context.init(allocator, block);
    defer ctx.deinit();

    try generate_prologue(&ctx);

    for (block.nodes.items) |node| {
        try generate_node(node, &ctx);
    }

    try generate_epilogue(&ctx);

    return ctx.code.toOwnedSlice();
}

fn generate_node(node: ir.Node, ctx: *Context) !void {
    switch (node.op) {
        .DEFINE_GLOBAL => {},
        .DEFINE_ACC => try generate_define_acc(node, ctx),
        .CONST => try generate_const(node, ctx),
        .LOOP => try generate_loop(node, ctx),
        //.LOAD => try generate_load(node, ctx),
        //.ALU => try generate_alu(node, ctx),
        //.UPDATE => try generate_update(node, ctx),
        .ENDLOOP => try generate_endloop(node, ctx),
        //.STORE => try generate_store(node, ctx),
        else => {},
    }
}

fn generate_define_acc(node: ir.Node, ctx: *Context) !void {
    const dest = ctx.registers.read();

    try mov(
        dest,
        .{ .Immediate = .{ .dtype = node.dtype.?, .value = node.arg.DEFINE_ACC } },
        ctx.code.writer(),
    );

    try ctx.map.put(node.step, dest);
}

const Mov_Options = union(enum) {
    Register: Register,
    Immediate: struct {
        dtype: ir.DataTypes,
        value: []const u8,
    },
};

fn mov(dest: Register, source: Mov_Options, writer: anytype) !void {
    switch (source) {
        .Register => |source_register| {
            std.debug.panic("mov: register-register unimplemented", .{});

            std.log.debug("mov {s}, {s}", .{ @tagName(dest), @tagName(source_register) });
        },
        .Immediate => |immediate| {
            var value: [4]u8 = undefined;

            try value_to_4_bytes(immediate.dtype, &value, immediate.value);

            if (mem.bytesToValue(u32, &value) == 0) {
                try (Instruction{
                    .op = .xor,
                    .dest = dest,
                    .source = .{ .Register = dest },
                }).encode(writer);

                std.log.debug("xor {s}, {s}", .{ @tagName(dest), @tagName(dest) });
            } else {
                try (Instruction{
                    .op = .mov_r64_imm,
                    .dest = dest,
                    .source = .{ .Immediate = &value },
                }).encode(writer);

                std.log.debug("mov {s}, {s}", .{ @tagName(dest), immediate.value });
            }
        },
    }
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
    try ctx.consts.put(node.step, node.arg.CONST);
}

fn generate_loop(node: ir.Node, ctx: *Context) !void {
    const dest = ctx.registers.read();

    const start_step = node.inputs.?[0];

    if (ctx.consts.get(start_step)) |value| {
        try mov(
            dest,
            .{ .Immediate = .{ .value = value, .dtype = .Int } },
            ctx.code.writer(),
        );
    }

    try ctx.map.put(node.step, dest);

    try ctx.labels.put(node.step, ctx.code.items.len);
}

fn generate_load(node: ir.Node, ctx: *Context) !void {
    _ = node;
    _ = ctx;
}

fn generate_alu(node: ir.Node, ctx: *Context) !void {
    _ = node;
    _ = ctx;
}

fn generate_update(node: ir.Node, ctx: *Context) !void {
    _ = node;
    _ = ctx;
}

fn generate_endloop(node: ir.Node, ctx: *Context) !void {
    const index = ctx.map.get(node.inputs.?[0]).?;

    try (Instruction{ .op = .inc, .source = .{ .Register = index } }).encode(ctx.code.writer());

    std.log.debug("inc {s}", .{@tagName(index)});

    if (ctx.map.get(ctx.block.nodes.items[node.inputs.?[0]].inputs.?[1])) |end| {
        try (Instruction{ .op = .cmp, .dest = index, .source = .{ .Register = end } }).encode(ctx.code.writer());

        std.log.debug("cmp {s}, {s}", .{ @tagName(index), @tagName(end) });
    } else if (ctx.consts.get(ctx.block.nodes.items[node.inputs.?[0]].inputs.?[1])) |immediate_value| {
        var value: [4]u8 = undefined;

        try value_to_4_bytes(.Int, &value, immediate_value);

        try (Instruction{ .op = .cmp_imm, .dest = index, .source = .{ .Immediate = &value } }).encode(ctx.code.writer());

        std.log.debug("cmp {s}, {s}", .{ @tagName(index), value });
    }
}

fn generate_store(node: ir.Node, ctx: *Context) !void {
    _ = node;
    _ = ctx;
}

fn generate_prologue(ctx: *Context) !void {
    _ = ctx;
}

fn generate_epilogue(ctx: *Context) !void {
    std.log.debug("ret", .{});
    try (Instruction{ .op = .ret }).encode(ctx.code.writer());
}
