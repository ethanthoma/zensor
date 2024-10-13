const std = @import("std");
const mem = std.mem;
const fmt = std.fmt;
const io = std.io;
const assert = std.debug.assert;

const ir = @import("../compiler/ir.zig");

// rbp: [*][*]u8
// rdi: function arg 1
// rsi: function arg 2

const Op = enum(u8) {
    ret = 0xc3,
    mov_r64_imm = 0xc7,
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

const Condition = enum(u4) { le };

const ModRM = struct {
    mod: u2,
    reg: u3,
    rm: u3,

    inline fn encode(self: @This()) u8 {
        const mod: u8 = self.mod;
        const reg: u8 = self.reg;
        const rm: u8 = self.rm;
        return (mod << 6) | (reg << 3) | rm;
    }
};

const SIB = struct {
    scale: u2,
    index: u3,
    base: u3,

    inline fn encode(self: @This()) u8 {
        const scale: u8 = self.scale;
        const index: u8 = self.index;
        const base: u8 = self.base;
        return (scale << 6) | (index << 3) | base;
    }
};

const Instruction = struct {
    op: Op,
    dest: ?Register = null,
    modrm: ?ModRM = null,
    immediate: ?[]const u8 = null,

    fn rex(self: @This()) ?u8 {
        var rex_byte: u8 = 0x40;

        if (self.dest) |dest| {
            if (dest.is_64_bit()) {
                rex_byte |= 0x48;
            }

            if (dest.is_extended()) {
                rex_byte |= 0x44;
            }
        }

        return if (rex_byte == 0x40) null else rex_byte;
    }

    fn encode(self: @This(), writer: anytype) !void {
        if (self.rex()) |rex_byte| {
            try writer.writeByte(rex_byte);
        }

        try writer.writeByte(@intFromEnum(self.op));

        if (self.modrm) |modrm| {
            try writer.writeByte(modrm.encode());
        }

        if (self.immediate) |immediate| {
            try writer.writeAll(immediate);
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
    code: std.ArrayList(u8),
    labels: std.AutoHashMap(ir.Step, usize),
    consts: std.AutoHashMap(ir.Step, []const u8),
    registers: CircularBuffer(Register),

    fn init(allocator: mem.Allocator) @This() {
        return .{
            .code = std.ArrayList(u8).init(allocator),
            .labels = std.AutoHashMap(ir.Step, usize).init(allocator),
            .consts = std.AutoHashMap(ir.Step, []const u8).init(allocator),
            .registers = CircularBuffer(Register).init(),
        };
    }

    fn deinit(self: *@This()) void {
        self.code.deinit();
        self.labels.deinit();
        self.consts.deinit();
    }
};

pub fn generate_kernel(allocator: mem.Allocator, block: *const ir.Block) ![]const u8 {
    var ctx = Context.init(allocator);
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
        //.LOOP => try generate_loop(node, ctx),
        //.LOAD => try generate_load(node, ctx),
        //.ALU => try generate_alu(node, ctx),
        //.UPDATE => try generate_update(node, ctx),
        //.ENDLOOP => try generate_endloop(node, ctx),
        //.STORE => try generate_store(node, ctx),
        else => {},
    }
}

fn generate_define_acc(node: ir.Node, ctx: *Context) !void {
    const reg = ctx.registers.read();

    var immediate: [4]u8 = undefined;

    try value_to_4_bytes(node.dtype.?, &immediate, node.arg.DEFINE_ACC);

    try (Instruction{
        .op = .mov_r64_imm,
        .dest = reg,
        .modrm = .{ .mod = 3, .reg = 0, .rm = reg.encoding() },
        .immediate = &immediate,
    }).encode(ctx.code.writer());

    std.debug.print("mov {s}, {s}", .{ @tagName(reg), node.arg.DEFINE_ACC });
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
    _ = node;
    _ = ctx;
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
    _ = node;
    _ = ctx;
}

fn generate_store(node: ir.Node, ctx: *Context) !void {
    _ = node;
    _ = ctx;
}

fn generate_prologue(ctx: *Context) !void {
    _ = ctx;
}

fn generate_epilogue(ctx: *Context) !void {
    try (Instruction{ .op = .ret }).encode(ctx.code.writer());
}
