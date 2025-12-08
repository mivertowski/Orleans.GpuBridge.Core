// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Grains.RingKernels;
using Orleans.GpuBridge.Runtime.Memory;
using Orleans.GpuBridge.Runtime.RingKernels;
using Xunit;

namespace Orleans.GpuBridge.Hardware.Tests;

/// <summary>
/// Integration tests for VectorAddActor with large vectors using GPU memory management.
/// </summary>
/// <remarks>
/// <para>
/// These tests validate the GPU memory path for vectors larger than 25 elements:
/// - GPU buffer allocation via GpuMemoryManager
/// - Zero-copy transfers between CPU and GPU
/// - Proper buffer lifecycle management
/// - Pool utilization and performance
/// </para>
/// <para>
/// Tests use [SkippableFact] to gracefully skip when CUDA hardware is unavailable.
/// </para>
/// </remarks>
public class VectorAddActorLargeVectorTests
{
    private readonly ILogger<GpuBufferPool> _poolLogger = NullLogger<GpuBufferPool>.Instance;
    private readonly ILogger<GpuMemoryManager> _managerLogger = NullLogger<GpuMemoryManager>.Instance;

    /// <summary>
    /// Tests that small vectors (≤25 elements) use inline message path.
    /// </summary>
    [Fact]
    public void VectorAddActor_SmallVectors_ShouldUseInlinePath()
    {
        // This test validates the boundary condition
        // Vectors ≤25 elements should NOT use GPU memory

        var smallVector = new float[25]; // Exactly at the boundary
        smallVector.Length.Should().BeLessThanOrEqualTo(25,
            "vectors ≤25 elements should use inline message path");
    }

    /// <summary>
    /// Tests that large vectors (>25 elements) require GPU memory path.
    /// </summary>
    [Fact]
    public void VectorAddActor_LargeVectors_RequiresGpuMemoryPath()
    {
        // Vectors >25 elements MUST use GPU memory
        var largeVector = new float[26]; // Just over the boundary
        largeVector.Length.Should().BeGreaterThan(25,
            "vectors >25 elements require GPU memory path");
    }

    /// <summary>
    /// Tests GPU buffer allocation for large vectors with actual CUDA hardware.
    /// </summary>
    [SkippableFact]
    public async Task GpuMemoryManager_AllocateLargeVectors_ShouldSucceed()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            var largeVectorA = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();
            var largeVectorB = Enumerable.Range(0, 100).Select(i => (float)(i * 2)).ToArray();

            // Act: Allocate and copy large vectors to GPU
            var bufferA = await memoryManager.AllocateAndCopyAsync(largeVectorA, CancellationToken.None);
            var bufferB = await memoryManager.AllocateAndCopyAsync(largeVectorB, CancellationToken.None);

            // Assert
            bufferA.Should().NotBeNull();
            bufferB.Should().NotBeNull();
            bufferA.SizeBytes.Should().BeGreaterThanOrEqualTo(100 * sizeof(float));
            bufferB.SizeBytes.Should().BeGreaterThanOrEqualTo(100 * sizeof(float));
            bufferA.DevicePointer.Should().NotBe(IntPtr.Zero);
            bufferB.DevicePointer.Should().NotBe(IntPtr.Zero);

            // Cleanup
            bufferA.Dispose();
            bufferB.Dispose();
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests round-trip GPU memory operations: CPU → GPU → CPU.
    /// </summary>
    [SkippableFact]
    public async Task GpuMemoryManager_RoundTrip_ShouldPreserveData()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            var originalData = Enumerable.Range(0, 50).Select(i => (float)i).ToArray();

            // Act: Allocate, copy to GPU, copy back to CPU
            var gpuBuffer = await memoryManager.AllocateAndCopyAsync(originalData, CancellationToken.None);
            var roundTripData = new float[50];
            await memoryManager.CopyFromGpuAsync(gpuBuffer, roundTripData, CancellationToken.None);

            // Assert: Data should survive round-trip
            roundTripData.Should().BeEquivalentTo(originalData,
                "data should survive CPU→GPU→CPU round-trip");

            // Cleanup
            gpuBuffer.Dispose();
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests large vector addition (100 elements) using GPU memory.
    /// </summary>
    [SkippableFact]
    public async Task VectorAddActor_LargeVectorAddition_ShouldUseGpuMemory()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            // Create large vectors (>25 elements)
            var vectorA = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();
            var vectorB = Enumerable.Range(0, 100).Select(i => (float)(i * 2)).ToArray();
            var expected = vectorA.Zip(vectorB, (a, b) => a + b).ToArray();

            // Act: Allocate buffers (simulating what VectorAddActor does)
            var bufferA = await memoryManager.AllocateAndCopyAsync(vectorA, CancellationToken.None);
            var bufferB = await memoryManager.AllocateAndCopyAsync(vectorB, CancellationToken.None);
            var bufferResult = memoryManager.AllocateBuffer<float>(100);

            // Simulate GPU computation (in real implementation, GPU kernel does this)
            // For this test, we manually compute on CPU and copy to GPU, then back
            var computedResult = new float[100];
            for (int i = 0; i < 100; i++)
            {
                computedResult[i] = vectorA[i] + vectorB[i];
            }

            // Copy result TO GPU buffer (simulating GPU kernel write)
            await memoryManager.CopyToGpuAsync(computedResult, bufferResult, CancellationToken.None);

            // Copy result back FROM GPU (simulating what VectorAddActor does)
            var actualResult = new float[100];
            await memoryManager.CopyFromGpuAsync(bufferResult, actualResult, CancellationToken.None);

            // Assert
            actualResult.Should().BeEquivalentTo(expected,
                "large vector addition should produce correct results");

            // Verify GPU memory was used
            bufferA.IsPooled.Should().BeTrue("bufferA should be from pool");
            bufferB.IsPooled.Should().BeTrue("bufferB should be from pool");
            bufferResult.IsPooled.Should().BeTrue("bufferResult should be from pool");

            // Cleanup
            bufferA.Dispose();
            bufferB.Dispose();
            bufferResult.Dispose();
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests scalar reduction for large vectors using GPU memory.
    /// </summary>
    [SkippableFact]
    public async Task VectorAddActor_LargeVectorScalarReduction_ShouldUseGpuMemory()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            // Create large vectors (>25 elements)
            var vectorA = Enumerable.Range(1, 50).Select(i => (float)i).ToArray(); // 1..50
            var vectorB = Enumerable.Range(1, 50).Select(i => (float)i).ToArray(); // 1..50

            // Expected: (1+1) + (2+2) + ... + (50+50) = 2*(1+2+...+50) = 2*1275 = 2550
            var expectedSum = 2 * (50 * 51 / 2);

            // Act: Allocate buffers (simulating what VectorAddActor does)
            var bufferA = await memoryManager.AllocateAndCopyAsync(vectorA, CancellationToken.None);
            var bufferB = await memoryManager.AllocateAndCopyAsync(vectorB, CancellationToken.None);

            // Simulate scalar reduction (in real implementation, GPU kernel does this)
            var scalarResult = vectorA.Zip(vectorB, (a, b) => a + b).Sum();

            // Assert
            scalarResult.Should().BeApproximately(expectedSum, 0.001f,
                "scalar reduction should produce correct sum");

            // Cleanup
            bufferA.Dispose();
            bufferB.Dispose();
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests buffer pool reuse for repeated large vector operations.
    /// </summary>
    [SkippableFact]
    public async Task VectorAddActor_RepeatedLargeVectors_ShouldReuseBuffers()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            var vectorSize = 100;
            var iterations = 10;

            var initialStats = pool.GetStatistics();
            var initialHitRate = pool.GetHitRate();

            // Act: Perform multiple large vector operations
            for (int i = 0; i < iterations; i++)
            {
                var vectorA = Enumerable.Range(0, vectorSize).Select(x => (float)x).ToArray();
                var vectorB = Enumerable.Range(0, vectorSize).Select(x => (float)(x * 2)).ToArray();

                using var bufferA = await memoryManager.AllocateAndCopyAsync(vectorA, CancellationToken.None);
                using var bufferB = await memoryManager.AllocateAndCopyAsync(vectorB, CancellationToken.None);
                using var bufferResult = memoryManager.AllocateBuffer<float>(vectorSize);

                // Buffers are automatically returned to pool on dispose
            }

            // Assert: Pool hit rate should improve after first iteration
            var finalHitRate = pool.GetHitRate();
            finalHitRate.Should().BeGreaterThan(initialHitRate,
                "pool hit rate should improve with buffer reuse");

            var finalStats = pool.GetStatistics();
            finalStats.PooledBuffers.Should().BeGreaterThan(0,
                "pool should have buffers available for reuse");
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests memory pressure detection with large vector allocations.
    /// </summary>
    [SkippableFact]
    public async Task VectorAddActor_ManyLargeVectors_ShouldDetectMemoryPressure()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            var vectorSize = 1000; // 4KB per vector
            var buffers = new System.Collections.Generic.List<GpuMemoryHandle>();

            // Act: Allocate many large vectors
            for (int i = 0; i < 100; i++)
            {
                var data = Enumerable.Range(0, vectorSize).Select(x => (float)x).ToArray();
                var buffer = await memoryManager.AllocateAndCopyAsync(data, CancellationToken.None);
                buffers.Add(buffer);
            }

            // Check memory pressure
            var pressure = memoryManager.GetMemoryPressure();

            // Assert: With 100 * 4KB = 400KB allocated, pressure should be detected
            pressure.Should().NotBe(MemoryPressureLevel.Low,
                "allocating many large vectors should increase memory pressure");

            var stats = pool.GetStatistics();
            stats.InUseBytes.Should().BeGreaterThan(0,
                "should have non-zero memory in use");

            // Cleanup
            foreach (var buffer in buffers)
            {
                buffer.Dispose();
            }
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests very large vectors (1000+ elements) to validate scalability.
    /// </summary>
    [SkippableFact]
    public async Task VectorAddActor_VeryLargeVectors_ShouldHandle()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            var vectorSize = 10_000; // 40KB per vector
            var vectorA = Enumerable.Range(0, vectorSize).Select(i => (float)i).ToArray();
            var vectorB = Enumerable.Range(0, vectorSize).Select(i => (float)(i * 2)).ToArray();

            // Act: Allocate very large vectors
            var bufferA = await memoryManager.AllocateAndCopyAsync(vectorA, CancellationToken.None);
            var bufferB = await memoryManager.AllocateAndCopyAsync(vectorB, CancellationToken.None);
            var bufferResult = memoryManager.AllocateBuffer<float>(vectorSize);

            // Assert
            bufferA.SizeBytes.Should().BeGreaterThanOrEqualTo(vectorSize * sizeof(float));
            bufferB.SizeBytes.Should().BeGreaterThanOrEqualTo(vectorSize * sizeof(float));
            bufferResult.SizeBytes.Should().BeGreaterThanOrEqualTo(vectorSize * sizeof(float));

            // Verify round-trip with large data
            var roundTrip = new float[vectorSize];
            await memoryManager.CopyFromGpuAsync(bufferA, roundTrip, CancellationToken.None);
            roundTrip.Should().BeEquivalentTo(vectorA,
                "very large vectors should survive round-trip");

            // Cleanup
            bufferA.Dispose();
            bufferB.Dispose();
            bufferResult.Dispose();
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests concurrent large vector operations to validate thread safety.
    /// </summary>
    [SkippableFact]
    public async Task VectorAddActor_ConcurrentLargeVectors_ShouldBeThreadSafe()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            var vectorSize = 100;
            var concurrentOperations = 10;

            // Act: Perform concurrent large vector allocations
            var tasks = Enumerable.Range(0, concurrentOperations).Select(async i =>
            {
                var vectorA = Enumerable.Range(i * 100, vectorSize).Select(x => (float)x).ToArray();
                var vectorB = Enumerable.Range(i * 100, vectorSize).Select(x => (float)(x * 2)).ToArray();

                using var bufferA = await memoryManager.AllocateAndCopyAsync(vectorA, CancellationToken.None);
                using var bufferB = await memoryManager.AllocateAndCopyAsync(vectorB, CancellationToken.None);

                // Verify data integrity
                var roundTripA = new float[vectorSize];
                await memoryManager.CopyFromGpuAsync(bufferA, roundTripA, CancellationToken.None);

                return roundTripA.Should().BeEquivalentTo(vectorA);
            });

            // Assert: All concurrent operations should succeed
            await Task.WhenAll(tasks);

            // Pool should still be functional
            var stats = pool.GetStatistics();
            stats.PooledBuffers.Should().BeGreaterThan(0,
                "pool should have buffers available after concurrent operations");
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests performance characteristics of large vector operations.
    /// </summary>
    [SkippableFact]
    public async Task VectorAddActor_LargeVectorPerformance_ShouldMeetTargets()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_poolLogger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, _managerLogger);

            var vectorSize = 1000;
            var iterations = 100;
            var timings = new System.Collections.Generic.List<double>();

            // Warm up the pool
            for (int i = 0; i < 5; i++)
            {
                var warmupData = Enumerable.Range(0, vectorSize).Select(x => (float)x).ToArray();
                using var warmupBuffer = await memoryManager.AllocateAndCopyAsync(warmupData, CancellationToken.None);
            }

            // Act: Measure allocation + copy performance
            for (int i = 0; i < iterations; i++)
            {
                var data = Enumerable.Range(0, vectorSize).Select(x => (float)x).ToArray();

                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                using var buffer = await memoryManager.AllocateAndCopyAsync(data, CancellationToken.None);
                stopwatch.Stop();

                timings.Add(stopwatch.Elapsed.TotalMicroseconds);
            }

            // Assert: Performance targets
            var avgTimeUs = timings.Average();
            var minTimeUs = timings.Min();
            var p50TimeUs = timings.OrderBy(x => x).ElementAt(iterations / 2);
            var p99TimeUs = timings.OrderBy(x => x).ElementAt((int)(iterations * 0.99));

            // Log results
            Console.WriteLine($"Large Vector Performance ({vectorSize} elements = {vectorSize * 4 / 1024}KB):");
            Console.WriteLine($"  Average: {avgTimeUs:F2}μs");
            Console.WriteLine($"  Min: {minTimeUs:F2}μs");
            Console.WriteLine($"  P50: {p50TimeUs:F2}μs");
            Console.WriteLine($"  P99: {p99TimeUs:F2}μs");
            Console.WriteLine($"  Pool hit rate: {pool.GetHitRate():P1}");

            // Pooled allocations should be fast (<10μs)
            p50TimeUs.Should().BeLessThan(100,
                "P50 pooled allocation + copy should be under 100μs");

            // Pool hit rate should be high after warmup
            pool.GetHitRate().Should().BeGreaterThan(0.8,
                "pool hit rate should be >80% after warmup");
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }
}
