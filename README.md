<h3 align="center">
    Zensor, a zig tensor library
</h3>

A zig tensor library. Correctness first, speed second.

This library promises compile-time type and shape checking.

**Very WIP**

## Example Usage:
```zig 
const std = @import("std");

const T = u32;
const Tensor = @import("zensor").Tensor(T);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var compiler = zensor.Compiler.init(allocator);
    defer compiler.deinit();

    const filename = "./examples/numpy.npy";

    const a = try zensor.Tensor(.Int64, .{3}).from_numpy(&compiler, filename);

    const b = try zensor.Tensor(.Int64, .{3}).full(&compiler, 4);

    const c = try a.mul(b);

    const d = try c.sum(0);

    std.debug.print("{}\n", .{d});
}
```

Results in:
```
❯ zig build run
Tensor(
        type: dtypes.Int64,
        shape: [1],
        length: 1,
        data: [56, ]
)
```

## Install

Fetch the library:
```bash
zig fetch --save git+https://github.com/ethanthoma/zensor.git#main
```

Add to your `build.zig`:
```zig
    const zensor = b.dependency("zensor", .{
        .target = target,
        .optimize = optimize,
    }).module("zensor");
    exe.root_module.addImport("zensor", zensor);
```

## Examples

Examples can be found in `./examples`. You can run these via:
```bash
zig build NAME_OF_EXAMPLE
```
Assuming you have cloned the source.

## Tests

If you want to run the tests after cloning the source. Simply run:
```bash
zig build test
```

## Design

Let's take an example op:
```zig
const a = try zensor.Tensor(.Int64, .{3}).from_numpy(&compiler, filename); // [ 1, 4, 9 ]

const b = try zensor.Tensor(.Int64, .{3}).full(&compiler, 4); // [ 4, 4, 4 ]

const c = try a.mul(b); // [ 4, 16, 36 ]

std.debug.print("C = A.mul(B) = {}\n", .{c});
```

This library converts all tensor operations into an AST:
```
0 Store RuntimeBuffer(ptr=@139636592083200, dtype=dtypes.Int64, shape={ 3 })
1 ┗━Mul
2   ┣━Load RuntimeBuffer(ptr=@139636592082944, dtype=dtypes.Int64, shape={ 3 })
3   ┗━Const 4
```

When you want to execute your operations, like when you `print` the tensor or `realize` it,
the AST is split into schedules:
```
Schedule{
        status: NotRun
        topological sort: [4]ast.Nodes{Load, Const, Mul, Store},
        global buffers: [(0, true), (1, false)],
        dependencies count: 0,
        AST:
        0 Store RuntimeBuffer(ptr=@139636592083200, dtype=dtypes.Int64, shape={ 3 })
        1 ┗━Mul
        2   ┣━Load RuntimeBuffer(ptr=@139636592082944, dtype=dtypes.Int64, shape={ 3 })
        3   ┗━Const 4
}
```
This is done so that it will generate a single kernel if you don't need intermediate results.

The current status is `NotRun`. The idea is that if the buffers and kernel are the 
same (i.e., used in a previous step somewhere), we can just reuse the results and not rerun the kernel.

When running, we convert the AST into IR:
```
step op name          type             input            arg
   0 DEFINE_GLOBAL    Pointer          []               (0, true)
   1 DEFINE_GLOBAL    Pointer          []               (1, false)
   2 CONST            Int              []               4
   3 CONST            Int              []               0
   4 CONST            Int              []               3
   5 LOOP             Int              [3, 4]           None
   6 LOAD             Int              [1, 5]           None
   7 ALU              Int              [6, 2]           ALU.Mul
   8 STORE                             [0, 5, 7]        None
   9 ENDLOOP                           [5]              None
```
This IR is based on the tinygrad IR (at some point) but will likely be changed into full SSA.

We convert the IR into bytecode (only x86 atm):
```
push Rbp
mov Rbp, Rsp
push { 0, 0, 0, 0 }
label_0x9:
mov R8, qword ptr [Rdi + 0x8]
mov R9, qword ptr [Rbp + 0x-8]
mov R8, qword ptr [R8 + R9 * 8]
mov R10, { 4, 0, 0, 0 }
mov R11, R8
imul R11, R10
mov R8, qword ptr [Rdi + 0x0]
mov R9, qword ptr [Rbp + 0x-8]
mov qword ptr [R8 + R9 * 8], R11
mov R10, qword ptr [Rbp + 0x-8]
inc R10
mov qword ptr [Rbp + 0x-8], R10
mov R11, R11
mov R11, { 3, 0, 0, 0 }
cmp R10, R11
jl 0x9
leave
ret
```

And finally, executed:
```
Tensor(
        type: dtypes.Int64,
        shape: [3],
        length: 3,
        data: [4, 16, 36, ]
)
```
