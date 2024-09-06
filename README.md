<h3 align="center">
    Zensor, a zig tensor library
</h3>

A zig tensor library. Correctness first, speed second.

This 

**Very WIP**

- [x] Basic generic Tensor type for floats and ints
- [x] Creation functions (ones, zeroes, full)
- [x] Create fromOwnedSlice
- [x] Elementwise ops
- [x] Broadcasting
- [x] Matmul only for rank 2 tensors
- [ ] Matmul w/broadcasting
- [ ] Refactor files for easier management
- [ ] Save/loading (possible numpy data format integration)
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
zig build NAME_OF_EXAMPLE
```
Assuming you have cloned the source.

## Tests

If you want to run the tests after cloning the source. Simply run:
```bash
zig build test
```

## Performance

This is WIP library so perf sucks as that is not the target atm.

However, I did decide to try a somewhat optimized matmul. You can run the example
via `zig build benchmark -Doptimize=ReleaseFast`. It will yield:
```
Running benchmark:
        Tensor Type: f32
        Tensor matmul: two 512x512 Tensors
        Number of Trials: 1000
Validating...
Starting trials...
Progress: 1000/1000 (100.0%)
Total time: 8.239 s for 1000 trials.
Performance: 32.580 GFLOPS/s
```

Which is the performance I get on my AMD Ryzen 7 5700G.

The implementation is single core and uses random block sizes but does utilize 
SIMD. GFLOPS is probably super incorrect as I just take the naive 2 * SIZE^3 approach. 
Good enough for the shot-in-the-dark optimization.
