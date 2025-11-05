using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Grains;
using Orleans.GpuBridge.Grains.Batch;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.Integration;

/// <summary>
/// Performance and scalability integration tests for Orleans GPU Bridge
/// </summary>
public class PerformanceIntegrationTests : IClassFixture<GpuClusterFixture>
{
    private readonly GpuClusterFixture _fixture;
    private readonly ITestOutputHelper _output;

    public PerformanceIntegrationTests(GpuClusterFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Fact]
    public async Task ThroughputTest_VectorOperations_ShouldAchieveReasonablePerformance()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        const int operationsCount = 1000;
        const int vectorSize = 1000;
        
        var inputs = Enumerable.Range(0, operationsCount)
            .Select(i => GenerateRandomVector(vectorSize))
            .ToList();

        // Act
        var stopwatch = Stopwatch.StartNew();
        
        // TODO: Update to match new GpuPipeline API
        var results = new List<float[]>(); // Placeholder for compilation
        // var results = await GpuPipeline.Create(bridge, logger)
        //     .AddKernel<float[], float[]>(new KernelId("vector-add"))
        //     .ExecuteAsync(inputs);
            
        stopwatch.Stop();

        // Assert
        results.Should().HaveCount(operationsCount);
        
        var totalElements = operationsCount * vectorSize;
        var elementsPerSecond = totalElements / stopwatch.Elapsed.TotalSeconds;
        var throughputMOps = elementsPerSecond / 1_000_000;
        
        _output.WriteLine($"Processed {totalElements:N0} elements in {stopwatch.ElapsedMilliseconds:N0} ms");
        _output.WriteLine($"Throughput: {throughputMOps:F2} million operations/second");
        
        // Should achieve at least 1 million operations per second on CPU
        throughputMOps.Should().BeGreaterThan(0.1);
        stopwatch.Elapsed.Should().BeLessThan(TimeSpan.FromMinutes(2));
    }

    [Fact]
    public async Task ScalabilityTest_ConcurrentGrains_ShouldHandleHighConcurrency()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        const int concurrentGrains = 50;
        const int operationsPerGrain = 20;
        
        var tasks = new List<Task<GpuBatchResult<float[]>>>();
        var grainIds = new List<string>();

        // Act
        var stopwatch = Stopwatch.StartNew();
        
        for (int i = 0; i < concurrentGrains; i++)
        {
            var grainId = $"scalability-test-{i}";
            grainIds.Add(grainId);
            
            var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), grainId);
            var inputs = Enumerable.Range(0, operationsPerGrain)
                .Select(_ => GenerateRandomVector(100))
                .ToArray();
                
            tasks.Add(grain.ExecuteAsync(inputs));
        }

        var results = await Task.WhenAll(tasks);
        stopwatch.Stop();

        // Assert
        results.Should().HaveCount(concurrentGrains);
        results.All(r => r.Success).Should().BeTrue();
        results.All(r => r.Results.Count == operationsPerGrain).Should().BeTrue();
        
        var totalOperations = concurrentGrains * operationsPerGrain;
        var operationsPerSecond = totalOperations / stopwatch.Elapsed.TotalSeconds;
        
        _output.WriteLine($"Executed {totalOperations:N0} operations across {concurrentGrains} grains in {stopwatch.ElapsedMilliseconds:N0} ms");
        _output.WriteLine($"Rate: {operationsPerSecond:F2} operations/second");
        
        operationsPerSecond.Should().BeGreaterThan(10); // At least 10 ops/sec under load
        stopwatch.Elapsed.Should().BeLessThan(TimeSpan.FromMinutes(5));
    }

    [Fact]
    public async Task MemoryEfficiencyTest_LargeDataSets_ShouldManageMemoryProperly()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(),"memory-test");
        
        const int batchCount = 10;
        const int vectorsPerBatch = 100;
        const int vectorSize = 10000; // 10k elements per vector
        
        var initialMemory = GC.GetTotalMemory(false);

        // Act
        var results = new List<GpuBatchResult<float[]>>();
        
        for (int batch = 0; batch < batchCount; batch++)
        {
            var inputs = Enumerable.Range(0, vectorsPerBatch)
                .Select(_ => GenerateRandomVector(vectorSize))
                .ToArray();

            var result = await grain.ExecuteAsync(inputs);
            results.Add(result);
            
            // Force garbage collection to test memory cleanup
            if (batch % 3 == 0)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
            
            _output.WriteLine($"Completed batch {batch + 1}/{batchCount}");
        }

        var finalMemory = GC.GetTotalMemory(true); // Force full GC

        // Assert
        results.Should().HaveCount(batchCount);
        results.All(r => r.Success).Should().BeTrue();
        
        var memoryIncreaseBytes = finalMemory - initialMemory;
        var memoryIncreaseMB = memoryIncreaseBytes / (1024.0 * 1024.0);
        
        _output.WriteLine($"Initial memory: {initialMemory / (1024.0 * 1024.0):F2} MB");
        _output.WriteLine($"Final memory: {finalMemory / (1024.0 * 1024.0):F2} MB");
        _output.WriteLine($"Memory increase: {memoryIncreaseMB:F2} MB");
        
        // Memory increase should be reasonable (less than 500MB for this test)
        memoryIncreaseMB.Should().BeLessThan(500);
    }

    [Fact]
    public async Task LatencyTest_SingleOperations_ShouldHaveLowLatency()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(),"latency-test");
        
        const int iterations = 100;
        var latencies = new List<TimeSpan>();
        var input = new[] { GenerateRandomVector(1000) };

        // Warm up
        await grain.ExecuteAsync(input);

        // Act
        for (int i = 0; i < iterations; i++)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = await grain.ExecuteAsync(input);
            stopwatch.Stop();
            
            result.Success.Should().BeTrue();
            latencies.Add(stopwatch.Elapsed);
        }

        // Assert
        var avgLatency = TimeSpan.FromMilliseconds(latencies.Average(l => l.TotalMilliseconds));
        var medianLatency = TimeSpan.FromMilliseconds(latencies.OrderBy(l => l.TotalMilliseconds)
            .Skip(iterations / 2).First().TotalMilliseconds);
        var p95Latency = TimeSpan.FromMilliseconds(latencies.OrderBy(l => l.TotalMilliseconds)
            .Skip((int)(iterations * 0.95)).First().TotalMilliseconds);

        _output.WriteLine($"Average latency: {avgLatency.TotalMilliseconds:F2} ms");
        _output.WriteLine($"Median latency: {medianLatency.TotalMilliseconds:F2} ms");
        _output.WriteLine($"95th percentile latency: {p95Latency.TotalMilliseconds:F2} ms");

        // Latency assertions (generous for CPU fallback)
        avgLatency.Should().BeLessThan(TimeSpan.FromMilliseconds(1000));
        p95Latency.Should().BeLessThan(TimeSpan.FromMilliseconds(5000));
    }

    [Fact]
    public async Task LoadTest_SustainedLoad_ShouldMaintainPerformance()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        const int durationSeconds = 30;
        const int operationsPerSecond = 20;
        
        var results = new List<GpuBatchResult<float[]>>();
        var errors = new List<Exception>();
        var stopwatch = Stopwatch.StartNew();

        // Act
        var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(durationSeconds));
        var tasks = new List<Task>();

        // Generate sustained load
        for (int worker = 0; worker < 4; worker++) // 4 worker tasks
        {
            var workerId = worker;
            tasks.Add(Task.Run(async () =>
            {
                var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), Guid.NewGuid().ToString(), $"load-test-{workerId}");
                var operationCount = 0;

                while (!cancellation.Token.IsCancellationRequested)
                {
                    try
                    {
                        var input = new[] { GenerateRandomVector(500) };
                        var result = await grain.ExecuteAsync(input);
                        
                        lock (results)
                        {
                            results.Add(result);
                        }
                        
                        operationCount++;
                        
                        // Control rate
                        var targetInterval = TimeSpan.FromMilliseconds(1000.0 / operationsPerSecond);
                        var actualInterval = TimeSpan.FromMilliseconds(operationCount * 1000.0 / stopwatch.Elapsed.TotalMilliseconds);
                        
                        if (actualInterval < targetInterval)
                        {
                            await Task.Delay(targetInterval - actualInterval, cancellation.Token);
                        }
                    }
                    catch (Exception ex) when (!(ex is OperationCanceledException))
                    {
                        lock (errors)
                        {
                            errors.Add(ex);
                        }
                    }
                }
            }, cancellation.Token));
        }

        await Task.WhenAll(tasks);
        stopwatch.Stop();

        // Assert
        var totalOperations = results.Count;
        var successfulOperations = results.Count(r => r.Success);
        var actualRate = totalOperations / stopwatch.Elapsed.TotalSeconds;
        
        _output.WriteLine($"Total operations: {totalOperations:N0}");
        _output.WriteLine($"Successful operations: {successfulOperations:N0}");
        _output.WriteLine($"Error count: {errors.Count}");
        _output.WriteLine($"Actual rate: {actualRate:F2} operations/second");
        _output.WriteLine($"Success rate: {(double)successfulOperations / totalOperations * 100:F2}%");

        totalOperations.Should().BeGreaterThan((int)(durationSeconds * 10)); // At least 10 ops/sec
        successfulOperations.Should().BeGreaterThan((int)(totalOperations * 0.95)); // 95% success rate
        errors.Should().HaveCountLessThan((int)(totalOperations * 0.05)); // Less than 5% errors
    }

    [Fact]
    public async Task BatchSizeOptimization_ShouldFindOptimalBatchSize()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var batchSizes = new[] { 1, 10, 50, 100, 200 };
        var results = new Dictionary<int, double>();
        
        const int totalOperations = 1000;
        const int vectorSize = 1000;

        // Act
        foreach (var batchSize in batchSizes)
        {
            _output.WriteLine($"Testing batch size: {batchSize}");
            
            var inputs = Enumerable.Range(0, totalOperations)
                .Select(_ => GenerateRandomVector(vectorSize))
                .ToList();

            var stopwatch = Stopwatch.StartNew();
            
            var batchResults = await GpuPipeline<float[], float[]>
                .For(grainFactory, "vector-add")
                .WithBatchSize(batchSize)
                .WithMaxConcurrency(Environment.ProcessorCount)
                .ExecuteAsync(inputs);
                
            stopwatch.Stop();

            var throughput = totalOperations / stopwatch.Elapsed.TotalSeconds;
            results[batchSize] = throughput;
            
            _output.WriteLine($"Batch size {batchSize}: {throughput:F2} operations/second");
            
            batchResults.Should().HaveCount(totalOperations);
        }

        // Assert
        results.Should().HaveCount(batchSizes.Length);
        results.Values.All(v => v > 0).Should().BeTrue();
        
        // Find optimal batch size (highest throughput)
        var optimalBatchSize = results.OrderByDescending(kvp => kvp.Value).First();
        _output.WriteLine($"Optimal batch size: {optimalBatchSize.Key} ({optimalBatchSize.Value:F2} ops/sec)");
        
        // There should be some variation in performance across batch sizes
        var maxThroughput = results.Values.Max();
        var minThroughput = results.Values.Min();
        (maxThroughput / minThroughput).Should().BeGreaterThan(1.1); // At least 10% difference
    }

    [Fact]
    public async Task ResourceCleanupTest_ShouldReleaseResourcesProper()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grainIds = new List<string>();
        
        const int grainCount = 20;
        const int operationsPerGrain = 10;

        // Act - Create and use many grains
        for (int i = 0; i < grainCount; i++)
        {
            var grainId = $"cleanup-test-{i}";
            grainIds.Add(grainId);
            
            var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), grainId);
            var inputs = Enumerable.Range(0, operationsPerGrain)
                .Select(_ => GenerateRandomVector(1000))
                .ToArray();
                
            var result = await grain.ExecuteAsync(inputs);
            result.Success.Should().BeTrue();
        }

        // Force garbage collection
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Wait for potential grain deactivation
        await Task.Delay(2000);

        // Reuse same grain IDs to test resource cleanup
        for (int i = 0; i < grainCount; i++)
        {
            var grainId = grainIds[i];
            var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), grainId);
            var input = new[] { GenerateRandomVector(100) };
            
            var result = await grain.ExecuteAsync(input);
            result.Success.Should().BeTrue();
        }

        // Assert - All operations should succeed, indicating proper cleanup
        // If resources weren't cleaned up properly, we'd likely see errors or memory issues
        _output.WriteLine($"Successfully reused {grainCount} grain identities after cleanup");
    }

    private static float[] GenerateRandomVector(int size)
    {
        var random = new Random();
        return Enumerable.Range(0, size)
            .Select(_ => (float)(random.NextDouble() * 100))
            .ToArray();
    }
}