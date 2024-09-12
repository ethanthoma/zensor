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

This library conversts all your tensor operations into an AST:
```
0 Store RuntimeBuffer(ptr=@140052063859008, dtype=dtypes.Int64, shape={ 3 })
1 ┗━Mul
2   ┣━Load RuntimeBuffer(ptr=@140052063858688, dtype=dtypes.Int64, shape={ 3 })
3   ┗━Const 4
```

When you want to execute your operations, it first gets split into schedules:
```
Schedule{
        status: NotRun
        topological sort: [4]ast.Nodes{Load, Const, Mul, Store},
        global buffers: [(0, true), (1, false)],
        dependencies count: 0,
        AST:
        0 Store RuntimeBuffer(ptr=@140052063859008, dtype=dtypes.Int64, shape={ 3 })
        1 ┗━Mul
        2   ┣━Load RuntimeBuffer(ptr=@140052063858688, dtype=dtypes.Int64, shape={ 3 })
        3   ┗━Const 4
}
```

And then IR code:
```
step op name          type             input            arg
   0 DEFINE_GLOBAL    Pointer          []               (0, true)
   1 DEFINE_GLOBAL    Pointer          []               (1, false)
   2 CONST            Int              []               0
   3 CONST            Int              []               3
   4 DEFINE_ACC       Int              [5]              0
   5 LOOP             Int              [2, 3]           None
   6 LOAD             Int              [1, 5]           None
   7 ALU              Int              [4, 6]           ALU.Add
   8 UPDATE           Int              [4, 7]           None
   9 ENDLOOP                           [5]              None
  10 CONST            Int              []               0
  11 STORE                             [0, 10, 4]       None
```

And finally, executed:
```
PC:   0
PC:   1
PC:   2 CONST: 0
PC:   3 CONST: 3
PC:   4 ACC: 0
PC:   5 LOOP: from 0 to 3
PC:   6 LOAD: 4 from buffer 1 at 0
PC:   7 ALU: Add(0, 4) = 4e0
PC:   8 UPDATE: value stored in step 4 to 4e0
PC:   6 LOAD: 16 from buffer 1 at 1
PC:   7 ALU: Add(4e0, 16) = 2e1
PC:   8 UPDATE: value stored in step 4 to 2e1
PC:   6 LOAD: 36 from buffer 1 at 2
PC:   7 ALU: Add(2e1, 36) = 5.6e1
PC:   8 UPDATE: value stored in step 4 to 5.6e1
PC:   9
PC:  10 CONST: 0
PC:  11 STORE: 5.6e1 into 0 at 0
```
