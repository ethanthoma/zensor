**Very WIP**

Basic tensor library implemented in zig. Correctness first, speed second.

## Example Usage:
```zig 
const Tensor = @import("zensor").Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var A = try Tensor(u64).full(allocator, .{2, 3}, 4);
    defer A.deinit();
    std.debug.print("{}\n", .{A});

    const slice: []const []const u64 = &.{ &.{ 3, 2, 1 }, &.{ 3, 2, 1 } };
    var B = try Tensor(u64).fromOwnedSlice(allocator, slice);
    defer B.deinit();
    std.debug.print("{}\n", .{B});

    const C = try A.add(&B);
    defer C.deinit();
    std.debug.print("{}\n", .{C});
}
```

Results in:
```
❯ zig build run
Tensor(
        type: u64,
        shape: [2, 3],
        length: 6,
        data:
        [
                [4, 4, 4],
                [4, 4, 4]
        ]
)
Tensor(
        type: u64,
        shape: [2, 3],
        length: 6,
        data:
        [
                [3, 2, 1],
                [3, 2, 1]
        ]
)
Tensor(
        type: u64,
        shape: [2, 3],
        length: 6,
        data:
        [
                [7, 6, 5],
                [7, 6, 5]
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
