// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using Orleans.GpuBridge.Benchmarks;

// Run benchmarks
var config = DefaultConfig.Instance
    .WithOptions(ConfigOptions.DisableOptimizationsValidator);

Console.WriteLine("Orleans.GpuBridge Performance Benchmarks");
Console.WriteLine("========================================");
Console.WriteLine();
Console.WriteLine("Select benchmark suite:");
Console.WriteLine("1. Memory Pool Benchmarks");
Console.WriteLine("2. Serialization Benchmarks");
Console.WriteLine("3. Kernel Execution Benchmarks");
Console.WriteLine("4. Ring Buffer Benchmarks");
Console.WriteLine("5. End-to-End Benchmarks");
Console.WriteLine("6. Load Testing (NBomber)");
Console.WriteLine("0. Run all benchmarks");
Console.WriteLine();
Console.Write("Enter selection: ");

var selection = Console.ReadLine();

switch (selection)
{
    case "1":
        BenchmarkRunner.Run<MemoryPoolBenchmarks>(config);
        break;
    case "2":
        BenchmarkRunner.Run<SerializationBenchmarks>(config);
        break;
    case "3":
        BenchmarkRunner.Run<KernelExecutionBenchmarks>(config);
        break;
    case "4":
        BenchmarkRunner.Run<RingBufferBenchmarks>(config);
        break;
    case "5":
        BenchmarkRunner.Run<EndToEndBenchmarks>(config);
        break;
    case "6":
        LoadTestRunner.RunLoadTests();
        break;
    case "0":
        BenchmarkRunner.Run<MemoryPoolBenchmarks>(config);
        BenchmarkRunner.Run<SerializationBenchmarks>(config);
        BenchmarkRunner.Run<KernelExecutionBenchmarks>(config);
        BenchmarkRunner.Run<RingBufferBenchmarks>(config);
        BenchmarkRunner.Run<EndToEndBenchmarks>(config);
        break;
    default:
        Console.WriteLine("Invalid selection");
        break;
}

Console.WriteLine();
Console.WriteLine("Benchmarks completed. Press any key to exit.");
Console.ReadKey();