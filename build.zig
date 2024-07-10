const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zensorModule = b.addModule("zensor", .{
        .root_source_file = b.path("src/zensor.zig"),
    });

    const exe = b.addExecutable(.{
        .name = "zensor.zig example",
        .root_source_file = b.path("example/example.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zensor", zensorModule);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);

    const tests = b.addTest(.{
        .root_source_file = b.path("src/zensor.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_test = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_test.step);
}
