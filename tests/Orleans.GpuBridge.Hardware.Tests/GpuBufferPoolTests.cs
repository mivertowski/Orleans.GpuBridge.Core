// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using System;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Backends.CUDA;
using DotCompute.Backends.CUDA.Memory;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Memory;
using Xunit;

namespace Orleans.GpuBridge.Hardware.Tests;

/// <summary>
/// Integration tests for GpuBufferPool with actual CUDA memory allocation.
/// Tests verify DotCompute integration, unified memory, and pool statistics.
/// </summary>
public class GpuBufferPoolTests
{
    private readonly ILogger<GpuBufferPool> _logger = NullLogger<GpuBufferPool>.Instance;

    /// <summary>
    /// Tests that GpuBufferPoolFactory.CreateAuto() successfully creates a pool.
    /// Should use GPU mode if CUDA available, otherwise falls back to CPU mode.
    /// </summary>
    [Fact]
    public void GpuBufferPoolFactory_CreateAuto_ShouldSucceed()
    {
        // Act
        using var pool = GpuBufferPoolFactory.CreateAuto(_logger);

        // Assert
        pool.Should().NotBeNull();
        var stats = pool.GetStatistics();
        stats.TotalAllocatedBytes.Should().Be(0, "pool starts empty");
    }

    /// <summary>
    /// Tests GPU-only mode allocation (requires CUDA hardware).
    /// Skipped if CUDA is not available on the system.
    /// </summary>
    [SkippableFact]
    public void GpuBufferPoolFactory_CreateGpuMode_WithCuda_ShouldSucceed()
    {
        // Arrange
        try
        {
            // Act
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_logger, deviceId: 0);

            // Assert
            pool.Should().NotBeNull();
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests CPU fallback mode (should always work, no CUDA required).
    /// </summary>
    [Fact]
    public void GpuBufferPoolFactory_CreateCpuFallbackMode_ShouldSucceed()
    {
        // Act
        using var pool = GpuBufferPoolFactory.CreateCpuFallbackMode(_logger);

        // Assert
        pool.Should().NotBeNull();
        var stats = pool.GetStatistics();
        stats.TotalAllocatedBytes.Should().Be(0);
    }

    /// <summary>
    /// Tests basic buffer allocation and deallocation in CPU fallback mode.
    /// </summary>
    [Fact]
    public void GpuBufferPool_RentAndReturn_CpuMode_ShouldWork()
    {
        // Arrange
        using var pool = GpuBufferPoolFactory.CreateCpuFallbackMode(_logger);

        // Act
        var buffer = pool.RentBuffer(1024); // 1 KB
        buffer.Should().NotBeNull();
        buffer.SizeBytes.Should().BeGreaterThanOrEqualTo(1024);
        buffer.DevicePointer.Should().NotBe(IntPtr.Zero);

        var stats1 = pool.GetStatistics();
        stats1.ActiveAllocations.Should().Be(1);

        buffer.Dispose();

        var stats2 = pool.GetStatistics();
        stats2.ActiveAllocations.Should().Be(0);
    }

    /// <summary>
    /// Tests GPU unified memory allocation with DotCompute CudaMemoryManager.
    /// Requires CUDA hardware - skipped if not available.
    /// </summary>
    [SkippableFact]
    public void GpuBufferPool_RentBuffer_GpuMode_ShouldAllocateUnifiedMemory()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_logger, deviceId: 0);

            // Act
            using var buffer = pool.RentBuffer(1024 * 1024); // 1 MB

            // Assert
            buffer.Should().NotBeNull();
            buffer.SizeBytes.Should().BeGreaterThanOrEqualTo(1024 * 1024);
            buffer.DevicePointer.Should().NotBe(IntPtr.Zero);

            var stats = pool.GetStatistics();
            stats.ActiveAllocations.Should().Be(1);
            stats.InUseBytes.Should().BeGreaterThanOrEqualTo(1024 * 1024);
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests pool hit rate statistics after multiple allocations.
    /// First allocation is a miss, subsequent same-size allocations should be hits.
    /// </summary>
    [Fact]
    public void GpuBufferPool_PoolHitRate_ShouldTrackReuse()
    {
        // Arrange
        using var pool = GpuBufferPoolFactory.CreateCpuFallbackMode(_logger);

        // Act - First allocation (pool miss)
        using (var buffer1 = pool.RentBuffer(1024))
        {
            buffer1.Should().NotBeNull();
        }

        // First allocation always a miss
        var hitRate1 = pool.GetHitRate();
        hitRate1.Should().Be(0.0, "first allocation is always a pool miss");

        // Second allocation (pool hit - buffer returned to pool)
        using (var buffer2 = pool.RentBuffer(1024))
        {
            buffer2.Should().NotBeNull();
        }

        var hitRate2 = pool.GetHitRate();
        hitRate2.Should().BeGreaterThan(0.0, "second allocation should be a pool hit");
    }

    /// <summary>
    /// Tests buffer reference counting mechanism.
    /// Multiple references to same buffer should work correctly.
    /// </summary>
    [Fact]
    public void GpuMemoryHandle_ReferenceCountingShouldWork()
    {
        // Arrange
        using var pool = GpuBufferPoolFactory.CreateCpuFallbackMode(_logger);
        var buffer = pool.RentBuffer(1024);

        // Act
        var refCount1 = buffer.ReferenceCount;
        var buffer2 = buffer.AddReference();
        var refCount2 = buffer.ReferenceCount;

        // Assert
        refCount1.Should().Be(1, "single reference initially");
        refCount2.Should().Be(2, "two references after AddReference()");
        buffer2.Should().BeSameAs(buffer, "AddReference returns same instance");

        // Cleanup
        buffer.Dispose();
        buffer2.Dispose();
    }

    /// <summary>
    /// Tests GPU memory manager integration with copy operations.
    /// Requires CUDA hardware - skipped if not available.
    /// </summary>
    [SkippableFact]
    public async Task GpuMemoryManager_CopyToGpu_ShouldWork()
    {
        try
        {
            // Arrange
            using var pool = GpuBufferPoolFactory.CreateGpuMode(_logger, deviceId: 0);
            using var memoryManager = new GpuMemoryManager(pool, NullLogger<GpuMemoryManager>.Instance);

            var sourceData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

            // Act
            var buffer = await memoryManager.AllocateAndCopyAsync(sourceData, CancellationToken.None);
            buffer.Should().NotBeNull();
            buffer.SizeBytes.Should().BeGreaterThanOrEqualTo(sourceData.Length * sizeof(float));

            // Verify round-trip
            var resultData = new float[sourceData.Length];
            await memoryManager.CopyFromGpuAsync(buffer, resultData, CancellationToken.None);

            // Assert
            resultData.Should().BeEquivalentTo(sourceData, "data should survive GPU round-trip");

            // Cleanup
            buffer.Dispose();
        }
        catch (Exception ex) when (ex.Message.Contains("CUDA") || ex.Message.Contains("GPU"))
        {
            throw new SkipException($"CUDA not available: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests memory pressure detection and reporting.
    /// </summary>
    [Fact]
    public void GpuMemoryManager_GetMemoryPressure_ShouldReturnCorrectLevel()
    {
        // Arrange
        using var pool = GpuBufferPoolFactory.CreateCpuFallbackMode(_logger);
        using var memoryManager = new GpuMemoryManager(pool, NullLogger<GpuMemoryManager>.Instance);

        // Act (empty pool)
        var pressure1 = memoryManager.GetMemoryPressure();

        // Assert (no allocations = low pressure)
        pressure1.Should().Be(MemoryPressureLevel.Low, "empty pool has low pressure");
    }

    /// <summary>
    /// Tests clearing the buffer pool and releasing all memory.
    /// </summary>
    [Fact]
    public void GpuBufferPool_Clear_ShouldReleaseAllMemory()
    {
        // Arrange
        using var pool = GpuBufferPoolFactory.CreateCpuFallbackMode(_logger);

        // Allocate and return buffers to pool
        for (int i = 0; i < 5; i++)
        {
            using var buffer = pool.RentBuffer(1024);
        }

        var statsBefore = pool.GetStatistics();
        statsBefore.PooledBuffers.Should().BeGreaterThan(0, "pool should have buffers");

        // Act
        pool.Clear();

        // Assert
        var statsAfter = pool.GetStatistics();
        statsAfter.PooledBuffers.Should().Be(0, "pool should be empty after Clear()");
        statsAfter.TotalAllocatedBytes.Should().Be(0, "all memory should be released");
    }
}
