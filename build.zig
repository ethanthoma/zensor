const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zensorModule = b.addModule(LIBRARY_NAME, .{
        .root_source_file = b.path("src/" ++ LIBRARY_NAME ++ ".zig"),
    });

    const cgModule = b.addModule("cg", .{
        .root_source_file = b.path("src/" ++ "cg" ++ ".zig"),
    });

    addTest(b, target, optimize, zensorModule);

    addExamples(b, target, optimize, cgModule);
}

fn addTest(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lib_mod: *std.Build.Module) void {
    const tests_step = b.step("test", "Run tests");

    inline for (TEST_NAMES) |TEST_NAME| {
        const tests = b.addTest(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(TEST_DIR ++ TEST_NAME ++ ".zig"),
        });
        tests.root_module.addImport(LIBRARY_NAME, lib_mod);
        const test_run = b.addRunArtifact(tests);
        tests_step.dependOn(&test_run.step);
    }

    b.default_step.dependOn(tests_step);
}

fn addExamples(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lib_mod: *std.Build.Module) void {
    inline for (EXAMPLE_NAMES) |EXAMPLE_NAME| {
        const example_step = b.step(EXAMPLE_NAME, "Run " ++ EXAMPLE_NAME ++ " example");
        const example = b.addExecutable(.{
            .name = EXAMPLE_NAME,
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(EXAMPLES_DIR ++ EXAMPLE_NAME ++ ".zig"),
        });
        example.root_module.addImport(LIBRARY_NAME, lib_mod);
        const example_run = b.addRunArtifact(example);
        example_step.dependOn(&example_run.step);

        b.installArtifact(example);
    }
}

const LIBRARY_NAME = "zensor";

const TEST_DIR = "test/";

const TEST_NAMES = &.{
    "test",
};

const EXAMPLES_DIR = "examples/";

const EXAMPLE_NAMES = &.{
    "basic",
    "benchmark",
    "bfloat16",
    "compute_graph",
    "ops",
};
