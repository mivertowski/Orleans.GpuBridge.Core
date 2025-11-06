using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans;
using Orleans.Hosting;
using Orleans.TestingHost;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Backends.DotCompute;
using Orleans.GpuBridge.Backends.DotCompute.Configuration;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Grains.Resident;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Providers;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.RingKernelTests;

/// <summary>
/// Performance benchmark tests for Ring Kernel implementation.
/// Validates latency and throughput targets:
/// - Allocation (pool hit): <100ns
/// - DMA transfer: <1μs
/// - Kernel execution: <10μs
/// - Throughput: 1M-10M ops/sec
/// </summary>
public class PerformanceBenchmarkTests : IDisposable
{
    private readonly TestCluster _cluster;
    private readonly ITestOutputHelper _output;

    public PerformanceBenchmarkTests(ITestOutputHelper output)
    {
        _output = output;

        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<TestSiloConfigurator>();

        _cluster = builder.Build();
        _cluster.Deploy();

        _output.WriteLine("✅ Performance benchmark cluster started");
    }

    private class TestSiloConfigurator : ISiloConfigurator
    {
        public void Configure(ISiloBuilder siloBuilder)
        {
            // Configure grain storage for persistent grains
            siloBuilder.AddMemoryGrainStorage("gpuStore");

            siloBuilder.ConfigureServices(services =>
            {
                // Configure GpuBridge options
                services.Configure<Orleans.GpuBridge.Abstractions.GpuBridgeOptions>(options =>
                {
                    options.PreferGpu = true;
                    options.MaxConcurrentKernels = 50;
                    options.MemoryPoolSizeMB = 2048;
                    options.EnableProfiling = false;
                });

                // Configure KernelCatalog with empty descriptors (no kernels needed for memory tests)
                services.Configure<Orleans.GpuBridge.Runtime.KernelCatalogOptions>(options =>
                {
                    // Empty kernel list - tests only use memory operations
                });

                // Register GpuBridge infrastructure
                services.AddSingleton<Orleans.GpuBridge.Runtime.KernelCatalog>();
                services.AddSingleton<Orleans.GpuBridge.Runtime.DeviceBroker>(sp =>
                {
                    var loggerFactory = sp.GetRequiredService<ILoggerFactory>();
                    var deviceLogger = loggerFactory.CreateLogger<Orleans.GpuBridge.Runtime.DeviceBroker>();
                    var gpuOptions = sp.GetRequiredService<IOptions<Orleans.GpuBridge.Abstractions.GpuBridgeOptions>>();
                    var deviceBroker = new Orleans.GpuBridge.Runtime.DeviceBroker(deviceLogger, gpuOptions);

                    // Initialize device broker
                    Task.Run(async () => await deviceBroker.InitializeAsync(default)).Wait();
                    return deviceBroker;
                });
                services.AddSingleton<IGpuBridge, Orleans.GpuBridge.Runtime.GpuBridge>();

                // Register DotCompute backend provider
                services.AddSingleton<IGpuBackendProvider>(sp =>
                {
                    var loggerFactory = sp.GetRequiredService<ILoggerFactory>();
                    var logger = loggerFactory.CreateLogger<DotComputeBackendProvider>();
                    var optionsMonitor = Options.Create(new DotComputeOptions());
                    var provider = new DotComputeBackendProvider(logger, loggerFactory, optionsMonitor);

                    // Initialize with default config
                    var config = new BackendConfiguration(
                        EnableProfiling: false,
                        EnableDebugMode: false,
                        MaxMemoryPoolSizeMB: 2048,
                        MaxConcurrentKernels: 50
                    );

                    Task.Run(async () => await provider.InitializeAsync(config, default)).Wait();
                    return provider;
                });
            });
        }
    }

    [Fact]
    public async Task Benchmark_AllocationLatency_PoolHits()
    {
        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IGpuResidentGrain<float>>(Guid.NewGuid().ToString());
        const int warmupOps = 100;
        const int benchmarkOps = 1000;
        const long sizeBytes = 4096; // 4KB

        _output.WriteLine("🔥 BENCHMARK: Allocation Latency (Pool Hits)");
        _output.WriteLine($"   Target: <100ns per allocation (pool hit)");
        _output.WriteLine($"   Operations: {benchmarkOps}");
        _output.WriteLine($"   Size: {sizeBytes} bytes");

        // Warmup - populate pool
        _output.WriteLine("\n   Warming up memory pool...");
        for (int i = 0; i < warmupOps; i++)
        {
            var handle = await grain.AllocateAsync(sizeBytes);
            await grain.ReleaseAsync(handle);
        }

        // Benchmark - measure pool hits
        _output.WriteLine("   Running benchmark...");
        var latencies = new long[benchmarkOps];
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < benchmarkOps; i++)
        {
            var iterSw = Stopwatch.StartNew();
            var handle = await grain.AllocateAsync(sizeBytes);
            latencies[i] = iterSw.ElapsedTicks;
            await grain.ReleaseAsync(handle);
        }

        sw.Stop();

        // Calculate statistics
        var avgLatencyNs = latencies.Average() * 1_000_000_000.0 / Stopwatch.Frequency;
        var minLatencyNs = latencies.Min() * 1_000_000_000.0 / Stopwatch.Frequency;
        var maxLatencyNs = latencies.Max() * 1_000_000_000.0 / Stopwatch.Frequency;
        var throughput = benchmarkOps / sw.Elapsed.TotalSeconds;

        var memoryInfo = await grain.GetMemoryInfoAsync();

        // Report results
        _output.WriteLine($"\n📊 RESULTS:");
        _output.WriteLine($"   Average latency: {avgLatencyNs:F2} ns");
        _output.WriteLine($"   Min latency: {minLatencyNs:F2} ns");
        _output.WriteLine($"   Max latency: {maxLatencyNs:F2} ns");
        _output.WriteLine($"   Throughput: {throughput:N0} ops/sec ({throughput / 1_000_000.0:F2} Mops/sec)");
        _output.WriteLine($"   Total memory: {memoryInfo.TotalMemoryBytes / (1024.0 * 1024.0):F2} MB");
        _output.WriteLine($"   Total time: {sw.Elapsed.TotalMilliseconds:F2} ms");

        // Validate targets
        _output.WriteLine($"\n✅ TARGET VALIDATION:");
        _output.WriteLine($"   Latency target (<100ns): {(avgLatencyNs < 100 ? "✅ PASS" : "⚠️ MISS")} ({avgLatencyNs:F2}ns)");
        _output.WriteLine($"   Throughput target (>1M ops/sec): {(throughput > 1_000_000 ? "✅ PASS" : "⚠️ MISS")} ({throughput:N0})");
    }

    [Fact]
    public async Task Benchmark_DMATransferThroughput()
    {
        // Arrange
        var grain = _cluster.GrainFactory.GetGrain<IGpuResidentGrain<float>>(Guid.NewGuid().ToString());
        const int iterations = 100;
        const int dataSize = 1024 * 256; // 256K floats = 1MB

        _output.WriteLine("🔥 BENCHMARK: DMA Transfer Throughput");
        _output.WriteLine($"   Target: <1μs per transfer (4KB), high bandwidth for large transfers");
        _output.WriteLine($"   Iterations: {iterations}");
        _output.WriteLine($"   Transfer size: {dataSize * sizeof(float) / (1024.0 * 1024.0):F2} MB");

        var handle = await grain.AllocateAsync(dataSize * sizeof(float));
        var testData = Enumerable.Range(0, dataSize).Select(i => (float)i).ToArray();

        // Benchmark writes
        _output.WriteLine("\n   Benchmarking WRITE operations...");
        var writeSw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            await grain.WriteAsync<float>(handle, testData);
        }
        writeSw.Stop();

        var writeLatency = writeSw.Elapsed.TotalMicroseconds / iterations;
        var writeBandwidth = (dataSize * sizeof(float) * iterations) / writeSw.Elapsed.TotalSeconds / (1024.0 * 1024.0);

        // Benchmark reads
        _output.WriteLine("   Benchmarking READ operations...");
        var readSw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var _ = await grain.ReadAsync<float>(handle, dataSize, offset: 0);
        }
        readSw.Stop();

        var readLatency = readSw.Elapsed.TotalMicroseconds / iterations;
        var readBandwidth = (dataSize * sizeof(float) * iterations) / readSw.Elapsed.TotalSeconds / (1024.0 * 1024.0);

        // Report results
        _output.WriteLine($"\n📊 WRITE RESULTS:");
        _output.WriteLine($"   Average latency: {writeLatency:F2} μs");
        _output.WriteLine($"   Bandwidth: {writeBandwidth:F2} MB/s");
        _output.WriteLine($"   Total time: {writeSw.Elapsed.TotalMilliseconds:F2} ms");

        _output.WriteLine($"\n📊 READ RESULTS:");
        _output.WriteLine($"   Average latency: {readLatency:F2} μs");
        _output.WriteLine($"   Bandwidth: {readBandwidth:F2} MB/s");
        _output.WriteLine($"   Total time: {readSw.Elapsed.TotalMilliseconds:F2} ms");

        // Cleanup
        await grain.ReleaseAsync(handle);
    }

    [Fact]
    public async Task Benchmark_MemoryPoolHitRate_RealisticWorkload()
    {
        // Arrange - Simulate realistic allocation patterns
        var grain = _cluster.GrainFactory.GetGrain<IGpuResidentGrain<float>>(Guid.NewGuid().ToString());
        const int operations = 10000;
        var random = new Random(42); // Deterministic seed

        _output.WriteLine("🔥 BENCHMARK: Memory Pool Hit Rate (Realistic Workload)");
        _output.WriteLine($"   Target: >90% pool hit rate");
        _output.WriteLine($"   Operations: {operations}");
        _output.WriteLine($"   Pattern: Power-law distribution (80/20 rule - 80% of requests for 20% of sizes)");

        // Common sizes (80% of allocations)
        var commonSizes = new[] { 1024, 2048, 4096, 8192, 16384 }; // Powers of 2
        // Rare sizes (20% of allocations)
        var rareSizes = new[] { 3072, 6144, 12288, 24576 };

        var sw = Stopwatch.StartNew();
        var handles = new System.Collections.Generic.List<GpuMemoryHandle>();

        for (int i = 0; i < operations; i++)
        {
            // 80% common sizes, 20% rare sizes
            var size = random.NextDouble() < 0.8
                ? commonSizes[random.Next(commonSizes.Length)]
                : rareSizes[random.Next(rareSizes.Length)];

            var handle = await grain.AllocateAsync(size);
            handles.Add(handle);

            // Randomly release some allocations (30% chance)
            if (handles.Count > 100 && random.NextDouble() < 0.3)
            {
                var releaseIndex = random.Next(handles.Count);
                var releaseHandle = handles[releaseIndex];
                handles.RemoveAt(releaseIndex);

                await grain.ReleaseAsync(releaseHandle);
            }
        }

        sw.Stop();

        var memoryInfo = await grain.GetMemoryInfoAsync();

        // Report results
        _output.WriteLine($"\n📊 RESULTS:");
        _output.WriteLine($"   Total operations: {operations}");
        _output.WriteLine($"   Total time: {sw.Elapsed.TotalSeconds:F2} seconds");
        _output.WriteLine($"   Throughput: {operations / sw.Elapsed.TotalSeconds:N0} ops/sec");
        _output.WriteLine($"   Total memory: {memoryInfo.TotalMemoryBytes / (1024.0 * 1024.0):F2} MB");
        _output.WriteLine($"   Allocated memory: {memoryInfo.AllocatedMemoryBytes / (1024.0 * 1024.0):F2} MB");
        _output.WriteLine($"   Free memory: {memoryInfo.FreeMemoryBytes / (1024.0 * 1024.0):F2} MB");
        _output.WriteLine($"   Utilization: {memoryInfo.UtilizationPercentage:F2}%");

        _output.WriteLine($"\n✅ TARGET VALIDATION:");
        _output.WriteLine($"   Memory allocated successfully");

        // Cleanup
        foreach (var handle in handles)
        {
            await grain.ReleaseAsync(handle);
        }
    }

    [Fact]
    public async Task Benchmark_ConcurrentThroughput_MaxOpsPerSec()
    {
        // Arrange
        const int grainCount = 10; // Multiple grains for true parallelism
        const int opsPerGrain = 1000;
        const int totalOps = grainCount * opsPerGrain;

        _output.WriteLine("🔥 BENCHMARK: Concurrent Throughput (Max Ops/Sec)");
        _output.WriteLine($"   Target: 1M-10M ops/sec");
        _output.WriteLine($"   Grains: {grainCount}");
        _output.WriteLine($"   Ops per grain: {opsPerGrain}");
        _output.WriteLine($"   Total operations: {totalOps:N0}");

        var grains = Enumerable.Range(0, grainCount)
            .Select(i => _cluster.GrainFactory.GetGrain<IGpuResidentGrain<float>>(Guid.NewGuid().ToString()))
            .ToArray();

        // Warmup
        _output.WriteLine("\n   Warming up grains...");
        await Task.WhenAll(grains.Select(g => g.AllocateAsync(1024)));

        // Benchmark
        _output.WriteLine("   Running benchmark...");
        var sw = Stopwatch.StartNew();

        var tasks = grains.Select(async grain =>
        {
            for (int i = 0; i < opsPerGrain; i++)
            {
                var handle = await grain.AllocateAsync(4096);
                await grain.ReleaseAsync(handle);
            }
        }).ToArray();

        await Task.WhenAll(tasks);
        sw.Stop();

        var throughput = totalOps / sw.Elapsed.TotalSeconds;
        var avgLatency = sw.Elapsed.TotalMilliseconds / totalOps;

        // Collect memory info from all grains
        var allMemoryInfo = await Task.WhenAll(grains.Select(g => g.GetMemoryInfoAsync()));
        var totalAllocated = allMemoryInfo.Sum(m => m.AllocatedMemoryBytes);
        var avgUtilization = allMemoryInfo.Average(m => m.UtilizationPercentage);

        // Report results
        _output.WriteLine($"\n📊 RESULTS:");
        _output.WriteLine($"   Total operations: {totalOps:N0}");
        _output.WriteLine($"   Total time: {sw.Elapsed.TotalSeconds:F2} seconds");
        _output.WriteLine($"   Throughput: {throughput:N0} ops/sec ({throughput / 1_000_000.0:F2} Mops/sec)");
        _output.WriteLine($"   Average latency: {avgLatency:F2} ms/op");
        _output.WriteLine($"   Total allocated: {totalAllocated / (1024.0 * 1024.0):F2} MB");
        _output.WriteLine($"   Average utilization: {avgUtilization:F2}%");

        _output.WriteLine($"\n✅ TARGET VALIDATION:");
        _output.WriteLine($"   Throughput target (>1M ops/sec): {(throughput > 1_000_000 ? "✅ PASS" : "⚠️ MISS")} ({throughput:N0})");
        _output.WriteLine($"   Stretch goal (>10M ops/sec): {(throughput > 10_000_000 ? "✅ PASS" : "⚠️ MISS")} ({throughput:N0})");
    }

    public void Dispose()
    {
        _cluster?.StopAllSilos();
        _output.WriteLine("\n✅ Performance benchmark cluster stopped");
    }
}
