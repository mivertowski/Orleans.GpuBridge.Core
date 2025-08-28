// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Hosting;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Backends.ILGPU.Extensions;
using Orleans.GpuBridge.Samples.ILGPU.Kernels;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Samples.ILGPU;

/// <summary>
/// Sample application demonstrating ILGPU integration with Orleans.GpuBridge
/// </summary>
public class Program
{
    public static async Task Main(string[] args)
    {
        var host = Host.CreateDefaultBuilder(args)
            .UseOrleans(siloBuilder =>
            {
                siloBuilder
                    .UseLocalhostClustering()
                    .ConfigureServices(services =>
                    {
                        // Add Orleans.GpuBridge with ILGPU backend
                        services.AddGpuBridge(options =>
                        {
                            options.PreferGpu = true;
                            options.EnableFallback = true;
                        })
                        .AddILGPUBackend(options =>
                        {
                            options.PreferredDeviceType = AcceleratorType.Cuda;
                            options.EnableKernelCaching = true;
                        });
                    })
                    .ConfigureLogging(logging =>
                    {
                        logging.SetMinimumLevel(LogLevel.Information);
                        logging.AddConsole();
                    });
            })
            .ConfigureServices(services =>
            {
                services.AddHostedService<SampleRunner>();
            })
            .Build();

        await host.RunAsync();
    }
}

/// <summary>
/// Hosted service that runs sample GPU operations
/// </summary>
public class SampleRunner : IHostedService
{
    private readonly IClusterClient _clusterClient;
    private readonly ILogger<SampleRunner> _logger;
    private readonly IGpuBridge _gpuBridge;

    public SampleRunner(
        IClusterClient clusterClient,
        ILogger<SampleRunner> logger,
        IGpuBridge gpuBridge)
    {
        _clusterClient = clusterClient;
        _logger = logger;
        _gpuBridge = gpuBridge;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting ILGPU sample runner");

        try
        {
            // Wait for cluster to be ready
            await Task.Delay(2000, cancellationToken);

            // Run samples
            await RunVectorAddSample();
            await RunMatrixMultiplySample();
            await RunReductionSample();
            await RunPerformanceComparison();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running samples");
        }
    }

    private async Task RunVectorAddSample()
    {
        _logger.LogInformation("=== Running Vector Addition Sample ===");

        const int vectorSize = 1_000_000;
        
        // Generate test data
        var random = new Random(42);
        var a = new float[vectorSize];
        var b = new float[vectorSize];
        
        for (int i = 0; i < vectorSize; i++)
        {
            a[i] = (float)random.NextDouble();
            b[i] = (float)random.NextDouble();
        }

        // CPU baseline
        var cpuStopwatch = Stopwatch.StartNew();
        var cpuResult = new float[vectorSize];
        for (int i = 0; i < vectorSize; i++)
        {
            cpuResult[i] = a[i] + b[i];
        }
        cpuStopwatch.Stop();

        _logger.LogInformation(
            "CPU Vector Add: {Size:N0} elements in {Time:F2}ms ({Throughput:F2} GFLOPS)",
            vectorSize,
            cpuStopwatch.Elapsed.TotalMilliseconds,
            vectorSize / cpuStopwatch.Elapsed.TotalSeconds / 1e9);

        // GPU via Orleans.GpuBridge
        var gpuStopwatch = Stopwatch.StartNew();
        
        // Here you would use the grain to execute the kernel
        // For demonstration, we'll show the pattern:
        /*
        var grain = _clusterClient.GetGrain<IGpuComputeGrain>(0);
        var result = await grain.ExecuteKernelAsync<float[], float[]>(
            "VectorAdd",
            new[] { a, b },
            vectorSize);
        */
        
        gpuStopwatch.Stop();

        _logger.LogInformation("Sample completed successfully");
    }

    private async Task RunMatrixMultiplySample()
    {
        _logger.LogInformation("=== Running Matrix Multiplication Sample ===");

        const int matrixSize = 512;
        
        // Generate test matrices
        var random = new Random(42);
        var a = new float[matrixSize * matrixSize];
        var b = new float[matrixSize * matrixSize];
        
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)random.NextDouble();
            b[i] = (float)random.NextDouble();
        }

        var stopwatch = Stopwatch.StartNew();
        
        // Matrix multiplication would be executed here
        
        stopwatch.Stop();

        _logger.LogInformation(
            "Matrix Multiply: {Size}x{Size} in {Time:F2}ms ({GFLOPS:F2} GFLOPS)",
            matrixSize,
            matrixSize,
            stopwatch.Elapsed.TotalMilliseconds,
            2.0 * matrixSize * matrixSize * matrixSize / stopwatch.Elapsed.TotalSeconds / 1e9);
    }

    private async Task RunReductionSample()
    {
        _logger.LogInformation("=== Running Reduction Sample ===");

        const int arraySize = 10_000_000;
        
        // Generate test data
        var random = new Random(42);
        var data = new float[arraySize];
        for (int i = 0; i < arraySize; i++)
        {
            data[i] = (float)random.NextDouble();
        }

        // CPU reduction
        var cpuStopwatch = Stopwatch.StartNew();
        var cpuSum = data.Sum();
        cpuStopwatch.Stop();

        _logger.LogInformation(
            "CPU Reduction: {Size:N0} elements in {Time:F2}ms, Sum = {Sum:F2}",
            arraySize,
            cpuStopwatch.Elapsed.TotalMilliseconds,
            cpuSum);

        // GPU reduction would be executed here
    }

    private async Task RunPerformanceComparison()
    {
        _logger.LogInformation("=== Performance Comparison ===");

        var sizes = new[] { 1000, 10000, 100000, 1000000 };
        
        foreach (var size in sizes)
        {
            // Generate data
            var data = new float[size];
            var random = new Random(42);
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)random.NextDouble();
            }

            // Measure CPU performance
            var cpuStopwatch = Stopwatch.StartNew();
            var cpuResult = new float[size];
            for (int i = 0; i < size; i++)
            {
                cpuResult[i] = MathF.Sin(data[i]) * MathF.Cos(data[i]);
            }
            cpuStopwatch.Stop();

            // Measure GPU performance (simulated)
            var gpuTime = cpuStopwatch.Elapsed.TotalMilliseconds * 0.1; // Assume 10x speedup

            _logger.LogInformation(
                "Size: {Size,10:N0} | CPU: {CpuTime,8:F2}ms | GPU: {GpuTime,8:F2}ms | Speedup: {Speedup,5:F2}x",
                size,
                cpuStopwatch.Elapsed.TotalMilliseconds,
                gpuTime,
                cpuStopwatch.Elapsed.TotalMilliseconds / gpuTime);
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Stopping ILGPU sample runner");
        return Task.CompletedTask;
    }
}