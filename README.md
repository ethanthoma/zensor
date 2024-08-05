<h3 align="center">
    Decision Transformer in Tinygrad
</h3>

Basic tensor library implemented in zig. Correctness first, speed second.

**Very WIP**

- [x] Basic generic Tensor type for floats and ints
- [x] Creation functions (ones, zeroes, full)
- [x] Create fromOwnedSlice
- [x] Elementwise ops
- [x] Broadcasting
- [ ] Matmul
- [ ] Save/loading (possible numpy data format integration)
- [ ] Refactor files for easier management
- [ ] Movement functions (squeeze, stack, permute, expand, etc)
- [ ] Op functions (max/min, sum, tril)
- [ ] Casting
- [ ] Casting
- [ ] Einsum
- [ ] Explore lazy ops vs comptime ops graph

## Example Usage:
```zig 
const std = @import("std");

const T = u32;
const Tensor = @import("zensor").Tensor(T);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var A = try Tensor.full(allocator, .{ 2, 3 }, 4);
    defer A.deinit();
    std.debug.print("{}\n", .{A});

    const slice: []const []const T = &.{ &.{ 3, 2, 1 }, &.{ 3, 2, 1 } };
    var B = try Tensor.fromOwnedSlice(allocator, slice);
    defer B.deinit();
    std.debug.print("{}\n", .{B});

    var C = try A.add(&B);
    defer C.deinit();
    std.debug.print("{}\n", .{C});

    const D = try C.add(3);
    std.debug.print("{}\n", .{D});
}
```

Results in:
```
❯ zig build run
Tensor(
        type: u32,
        shape: [2, 3],
        length: 6,
        data:
        [
                [4, 4, 4],
                [4, 4, 4]
        ]
)
Tensor(
        type: u32,
        shape: [2, 3],
        length: 6,
        data:
        [
                [3, 2, 1],
                [3, 2, 1]
        ]
)
Tensor(
        type: u32,
        shape: [2, 3],
        length: 6,
        data:
        [
                [7, 6, 5],
                [7, 6, 5]
        ]
)
Tensor(
        type: u32,
        shape: [2, 3],
        length: 6,
        data:
        [
                [10, 9, 8],
                [10, 9, 8]
        ]
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
zig build examples
```
Assuming you have cloned the source.

## Tests

If you want to run the tests after cloning the source. Simply run:
```bash
zig build test
```
