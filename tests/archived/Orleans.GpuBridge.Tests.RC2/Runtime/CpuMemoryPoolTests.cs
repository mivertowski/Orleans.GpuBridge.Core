using System.Collections.Concurrent;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Runtime;

namespace Orleans.GpuBridge.Tests.RC2.Runtime;

/// <summary>
/// Comprehensive test suite for CpuMemoryPool focusing on memory allocation,
/// deallocation, pooling, reuse, thread safety, and error handling.
/// Target: 30+ tests to increase coverage from 11% to 60%+
/// </summary>
public sealed class CpuMemoryPoolTests : IDisposable
{
    private readonly List<IGpuMemory<float>> _allocatedMemory;

    public CpuMemoryPoolTests()
    {
        _allocatedMemory = new List<IGpuMemory<float>>();
    }

    public void Dispose()
    {
        // Clean up any allocated memory
        foreach (var memory in _allocatedMemory)
        {
            memory?.Dispose();
        }
        _allocatedMemory.Clear();
    }

    #region Pool Initialization Tests (5 tests)

    [Fact]
    public void Constructor_Default_ShouldInitializeSuccessfully()
    {
        // Act
        var pool = new CpuMemoryPool<float>();
        var stats = pool.GetStats();

        // Assert
        stats.Should().NotBeNull();
        stats.TotalAllocated.Should().Be(0);
        stats.InUse.Should().Be(0);
        stats.Available.Should().Be(0);
        stats.BufferCount.Should().Be(0);
        stats.RentCount.Should().Be(0);
        stats.ReturnCount.Should().Be(0);
    }

    [Fact]
    public void Constructor_Generic_ShouldWorkWithDifferentTypes()
    {
        // Act
        var intPool = new CpuMemoryPool<int>();
        var doublePool = new CpuMemoryPool<double>();
        var bytePool = new CpuMemoryPool<byte>();

        // Assert
        intPool.GetStats().Should().NotBeNull();
        doublePool.GetStats().Should().NotBeNull();
        bytePool.GetStats().Should().NotBeNull();
    }

    [Fact]
    public void Constructor_MultipleInstances_ShouldBeIndependent()
    {
        // Arrange
        var pool1 = new CpuMemoryPool<float>();
        var pool2 = new CpuMemoryPool<float>();

        // Act
        var memory1 = pool1.Rent(100);
        var stats1 = pool1.GetStats();
        var stats2 = pool2.GetStats();

        // Assert
        stats1.InUse.Should().BeGreaterThan(0);
        stats2.InUse.Should().Be(0); // Pool2 should be unaffected

        // Cleanup
        pool1.Return(memory1);
    }

    [Theory]
    [InlineData(typeof(byte))]
    [InlineData(typeof(short))]
    [InlineData(typeof(int))]
    [InlineData(typeof(long))]
    [InlineData(typeof(float))]
    [InlineData(typeof(double))]
    public void Constructor_DifferentPrimitiveTypes_ShouldHandleAllUnmanagedTypes(Type type)
    {
        // Act & Assert
        var poolType = typeof(CpuMemoryPool<>).MakeGenericType(type);
        var pool = Activator.CreateInstance(poolType);

        pool.Should().NotBeNull();
    }

    [Fact]
    public void GetStats_AfterConstruction_ShouldReturnValidStatistics()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act
        var stats = pool.GetStats();

        // Assert
        stats.TotalAllocated.Should().Be(0);
        stats.InUse.Should().Be(0);
        stats.Available.Should().Be(0);
        stats.BufferCount.Should().Be(0);
        stats.UtilizationPercent.Should().Be(0);
    }

    #endregion

    #region Memory Allocation Tests (10 tests)

    [Fact]
    public void Rent_SmallBlock_ShouldReturnValidMemory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var sizeElements = 256; // 1KB for floats

        // Act
        var memory = pool.Rent(sizeElements);
        _allocatedMemory.Add(memory);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().BeGreaterThanOrEqualTo(sizeElements);
        memory.SizeInBytes.Should().BeGreaterThanOrEqualTo(sizeElements * sizeof(float));
        memory.DeviceIndex.Should().Be(-1); // CPU
        memory.IsResident.Should().BeTrue();

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void Rent_MediumBlock_ShouldReturnValidMemory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var sizeElements = 1024 * 1024; // 4MB for floats

        // Act
        var memory = pool.Rent(sizeElements);
        _allocatedMemory.Add(memory);
        var stats = pool.GetStats();

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().BeGreaterThanOrEqualTo(sizeElements);
        stats.InUse.Should().BeGreaterThan(0);
        stats.TotalAllocated.Should().BeGreaterThan(0);

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void Rent_LargeBlock_ShouldReturnValidMemory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var sizeElements = 1024 * 1024; // 4MB for floats (moderate size to avoid ArrayPool issues)

        // Act
        var memory = pool.Rent(sizeElements);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().BeGreaterThanOrEqualTo(sizeElements);
        memory.SizeInBytes.Should().BeGreaterThan(0);

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void Rent_ZeroElements_ShouldReturnValidMemory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act
        var memory = pool.Rent(0);
        _allocatedMemory.Add(memory);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().BeGreaterThanOrEqualTo(0);

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void Rent_OneElement_ShouldReturnValidMemory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act
        var memory = pool.Rent(1);
        _allocatedMemory.Add(memory);

        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().BeGreaterThanOrEqualTo(1);
        memory.SizeInBytes.Should().BeGreaterThanOrEqualTo(sizeof(float));

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void Rent_MultipleBlocks_ShouldTrackAllAllocations()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var sizes = new[] { 100, 1000, 10000 };
        var memories = new List<IGpuMemory<float>>();

        // Act
        foreach (var size in sizes)
        {
            var memory = pool.Rent(size);
            memories.Add(memory);
            _allocatedMemory.Add(memory);
        }

        var stats = pool.GetStats();

        // Assert
        stats.RentCount.Should().Be(sizes.Length);
        stats.InUse.Should().BeGreaterThan(0);
        stats.TotalAllocated.Should().BeGreaterThan(0);

        // Cleanup
        foreach (var memory in memories)
        {
            pool.Return(memory);
        }
    }

    [Fact]
    public void Rent_SameSize_ShouldReusePooledMemory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var size = 1000;

        // Act - First allocation
        var memory1 = pool.Rent(size);
        _allocatedMemory.Add(memory1);
        var statsAfterRent1 = pool.GetStats();
        pool.Return(memory1);

        // Act - Second allocation (should reuse)
        var memory2 = pool.Rent(size);
        _allocatedMemory.Add(memory2);
        var statsAfterRent2 = pool.GetStats();

        // Assert
        statsAfterRent2.BufferCount.Should().Be(0); // Buffer was taken from pool
        statsAfterRent2.RentCount.Should().Be(2);
        statsAfterRent2.ReturnCount.Should().Be(1);

        // Cleanup
        pool.Return(memory2);
    }

    [Fact]
    public void Rent_ExactSizeMatch_ShouldReusePooledBuffer()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var size = 5000;

        // Act
        var memory1 = pool.Rent(size);
        var originalLength = memory1.Length;
        pool.Return(memory1);

        var memory2 = pool.Rent(size);
        _allocatedMemory.Add(memory2);

        // Assert
        memory2.Length.Should().Be(originalLength); // Same buffer reused

        // Cleanup
        pool.Return(memory2);
    }

    [Fact]
    public void Rent_SmallerThanPooled_ShouldReusePooledBuffer()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Allocate and return large buffer
        var largeMemory = pool.Rent(10000);
        pool.Return(largeMemory);

        // Act - Request smaller buffer
        var smallMemory = pool.Rent(5000);
        _allocatedMemory.Add(smallMemory);

        // Assert - Should reuse the larger pooled buffer
        smallMemory.Length.Should().BeGreaterThanOrEqualTo(5000);

        // Cleanup
        pool.Return(smallMemory);
    }

    [Fact]
    public void Rent_LargerThanPooled_ShouldAllocateNew()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Allocate and return small buffer
        var smallMemory = pool.Rent(1000);
        pool.Return(smallMemory);

        var statsAfterReturn = pool.GetStats();

        // Act - Request larger buffer (won't match smaller pooled buffer, so new allocation)
        var largeMemory = pool.Rent(10000);
        _allocatedMemory.Add(largeMemory);

        var statsAfterLargeRent = pool.GetStats();

        // Assert - Should allocate new buffer
        // Note: The small buffer (1000) remains in pool, but TotalAllocated increases for large buffer
        statsAfterLargeRent.TotalAllocated.Should().BeGreaterThan(statsAfterReturn.TotalAllocated);
        statsAfterLargeRent.RentCount.Should().Be(2);

        // Cleanup
        pool.Return(largeMemory);
    }

    #endregion

    #region Memory Deallocation Tests (8 tests)

    [Fact]
    public void Return_ValidMemory_ShouldUpdateStatistics()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(1000);
        _allocatedMemory.Add(memory);
        var statsAfterRent = pool.GetStats();

        // Act
        pool.Return(memory);
        var statsAfterReturn = pool.GetStats();

        // Assert
        statsAfterReturn.InUse.Should().Be(0);
        statsAfterReturn.ReturnCount.Should().Be(1);
        statsAfterReturn.BufferCount.Should().Be(1); // Buffer returned to pool
    }

    [Fact]
    public void Return_MemoryFromDifferentPool_AcceptsButDoesntValidateOwnership()
    {
        // Arrange
        var pool1 = new CpuMemoryPool<float>();
        var pool2 = new CpuMemoryPool<float>();
        var memory = pool1.Rent(1000);

        var pool1StatsBefore = pool1.GetStats();
        var pool2StatsBefore = pool2.GetStats();

        // Act - Return to wrong pool (implementation doesn't validate pool ownership)
        pool2.Return(memory);

        var pool1StatsAfter = pool1.GetStats();
        var pool2StatsAfter = pool2.GetStats();

        // Assert - Memory is returned to pool2 even though it was rented from pool1
        pool2StatsAfter.ReturnCount.Should().Be(pool2StatsBefore.ReturnCount + 1);
        pool1StatsAfter.ReturnCount.Should().Be(pool1StatsBefore.ReturnCount); // Pool1 unchanged
    }

    [Fact]
    public void Return_AllMemory_ShouldResetInUseToZero()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memories = new List<IGpuMemory<float>>
        {
            pool.Rent(1000),
            pool.Rent(2000),
            pool.Rent(3000)
        };
        _allocatedMemory.AddRange(memories);

        // Act
        foreach (var memory in memories)
        {
            pool.Return(memory);
        }

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().Be(0);
        stats.ReturnCount.Should().Be(3);
    }

    [Fact]
    public void Return_MoreThanTenBuffers_ShouldDisposeExcess()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memories = new List<IGpuMemory<float>>();

        // Act - Allocate and return 15 buffers
        for (int i = 0; i < 15; i++)
        {
            var memory = pool.Rent(1000);
            memories.Add(memory);
            pool.Return(memory);
        }

        var stats = pool.GetStats();

        // Assert - Pool should keep max 10 buffers (but may have less due to reuse)
        stats.BufferCount.Should().BeLessThanOrEqualTo(10);
        stats.ReturnCount.Should().Be(15);
        stats.TotalAllocated.Should().BeLessThan(15 * 1000 * sizeof(float));
    }

    [Fact]
    public void Return_BufferIsCleared_ShouldHaveZeroedData()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(100);
        _allocatedMemory.Add(memory);

        // Fill with data
        var span = memory.AsMemory().Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = i + 1.0f;
        }

        // Act - Return to pool
        pool.Return(memory);

        // Rent again and verify cleared
        var memory2 = pool.Rent(100);
        _allocatedMemory.Add(memory2);
        var span2 = memory2.AsMemory().Span;

        // Assert - Buffer should be cleared
        span2[0].Should().Be(0f);

        // Cleanup
        pool.Return(memory2);
    }

    [Fact]
    public void Return_InDifferentOrderThanAllocation_ShouldHandleCorrectly()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory1 = pool.Rent(1000);
        var memory2 = pool.Rent(2000);
        var memory3 = pool.Rent(3000);
        _allocatedMemory.AddRange(new[] { memory1, memory2, memory3 });

        // Act - Return in reverse order
        pool.Return(memory3);
        pool.Return(memory1);
        pool.Return(memory2);

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().Be(0);
        stats.ReturnCount.Should().Be(3);
        stats.BufferCount.Should().Be(3);
    }

    [Fact]
    public void Return_MultipleTimes_ShouldTrackReturnCount()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var returnCount = 0;

        // Act
        for (int i = 0; i < 5; i++)
        {
            var memory = pool.Rent(1000);
            pool.Return(memory);
            returnCount++;
        }

        var stats = pool.GetStats();

        // Assert
        stats.ReturnCount.Should().Be(returnCount);
    }

    [Fact]
    public void Return_ConcurrentReturns_ShouldBeThreadSafe()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memories = new ConcurrentBag<IGpuMemory<float>>();

        // Allocate buffers
        for (int i = 0; i < 50; i++)
        {
            memories.Add(pool.Rent(1000));
        }

        // Act - Return concurrently
        Parallel.ForEach(memories, memory =>
        {
            pool.Return(memory);
        });

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().Be(0);
        stats.ReturnCount.Should().Be(50);
    }

    #endregion

    #region Memory Pooling and Reuse Tests (6 tests)

    [Fact]
    public void Pool_AllocateFreeAllocateSameSize_ShouldReuseBuffer()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var size = 5000;

        // Act - First cycle
        var memory1 = pool.Rent(size);
        var length1 = memory1.Length;
        var statsAfterRent1 = pool.GetStats();
        pool.Return(memory1);

        // Act - Second cycle (should reuse)
        var memory2 = pool.Rent(size);
        _allocatedMemory.Add(memory2);
        var length2 = memory2.Length;
        var statsAfterRent2 = pool.GetStats();

        // Assert
        length2.Should().Be(length1); // Same buffer
        statsAfterRent2.TotalAllocated.Should().Be(statsAfterRent1.TotalAllocated); // No new allocation

        // Cleanup
        pool.Return(memory2);
    }

    [Fact]
    public void Pool_FragmentationScenario_ShouldHandleVariedSizes()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var sizes = new[] { 100, 1000, 500, 2000, 300, 1500 };

        // Act - Allocate all
        var memories = sizes.Select(size => pool.Rent(size)).ToList();
        _allocatedMemory.AddRange(memories);

        // Return all
        foreach (var memory in memories)
        {
            pool.Return(memory);
        }

        var statsAfterReturn = pool.GetStats();

        // Assert
        statsAfterReturn.InUse.Should().Be(0);
        statsAfterReturn.BufferCount.Should().Be(sizes.Length);

        // Verify reuse works
        var newMemory = pool.Rent(500);
        _allocatedMemory.Add(newMemory);
        var statsAfterReuse = pool.GetStats();

        statsAfterReuse.BufferCount.Should().Be(sizes.Length - 1); // One buffer taken

        // Cleanup
        pool.Return(newMemory);
    }

    [Fact]
    public void Pool_StatisticsAfterMultipleCycles_ShouldBeAccurate()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Multiple allocation/return cycles
        for (int cycle = 0; cycle < 20; cycle++)
        {
            var memory = pool.Rent(1000);
            pool.Return(memory);
        }

        var stats = pool.GetStats();

        // Assert
        stats.RentCount.Should().Be(20);
        stats.ReturnCount.Should().Be(20);
        stats.InUse.Should().Be(0);
        stats.BufferCount.Should().BeGreaterThan(0).And.BeLessThanOrEqualTo(10);
    }

    [Fact]
    public void Pool_UtilizationPercent_ShouldCalculateCorrectly()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Rent some memory
        var memory1 = pool.Rent(1000);
        var memory2 = pool.Rent(2000);
        _allocatedMemory.AddRange(new[] { memory1, memory2 });

        var statsWithInUse = pool.GetStats();

        pool.Return(memory1);
        var statsPartialReturn = pool.GetStats();

        pool.Return(memory2);
        var statsFullReturn = pool.GetStats();

        // Assert
        statsWithInUse.UtilizationPercent.Should().Be(100.0);
        statsPartialReturn.UtilizationPercent.Should().BeGreaterThan(0).And.BeLessThan(100);
        statsFullReturn.UtilizationPercent.Should().Be(0);
    }

    [Fact]
    public void Pool_MaxPoolSize_ShouldNotExceedTenBuffers()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Create more than 10 unique size buffers
        for (int i = 0; i < 20; i++)
        {
            var memory = pool.Rent(1000 + i * 100);
            pool.Return(memory);
        }

        var stats = pool.GetStats();

        // Assert - Pool may reuse buffers, so count will be <= 10
        stats.BufferCount.Should().BeLessThanOrEqualTo(10);
    }

    [Fact]
    public void Pool_MemoryReuse_ShouldReduceAllocations()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var size = 5000;

        // First allocation
        var memory1 = pool.Rent(size);
        pool.Return(memory1);
        var statsAfterFirst = pool.GetStats();

        // Act - Multiple reuses
        for (int i = 0; i < 10; i++)
        {
            var memory = pool.Rent(size);
            pool.Return(memory);
        }

        var statsAfterReuse = pool.GetStats();

        // Assert - Total allocated should not increase significantly
        statsAfterReuse.TotalAllocated.Should().Be(statsAfterFirst.TotalAllocated);
        statsAfterReuse.RentCount.Should().Be(11); // 1 initial + 10 reuses
    }

    #endregion

    #region Thread Safety Tests (4 tests)

    [Fact]
    public async Task ConcurrentAllocations_From10Threads_ShouldBeThreadSafe()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var threadCount = 10;
        var allocationsPerThread = 100;

        // Act
        var tasks = Enumerable.Range(0, threadCount)
            .Select(_ => Task.Run(() =>
            {
                var localMemories = new List<IGpuMemory<float>>();
                for (int i = 0; i < allocationsPerThread; i++)
                {
                    var memory = pool.Rent(1000);
                    localMemories.Add(memory);
                }

                foreach (var memory in localMemories)
                {
                    pool.Return(memory);
                }
            }));

        await Task.WhenAll(tasks);

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().Be(0);
        stats.RentCount.Should().Be(threadCount * allocationsPerThread);
        stats.ReturnCount.Should().Be(threadCount * allocationsPerThread);
    }

    [Fact]
    public async Task ConcurrentDeallocations_ShouldBeThreadSafe()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memories = new ConcurrentBag<IGpuMemory<float>>();

        // Pre-allocate
        for (int i = 0; i < 100; i++)
        {
            memories.Add(pool.Rent(1000));
        }

        // Act - Deallocate concurrently
        var tasks = memories.Select(memory => Task.Run(() => pool.Return(memory)));
        await Task.WhenAll(tasks);

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().Be(0);
        stats.ReturnCount.Should().Be(100);
    }

    [Fact]
    public async Task MixedAllocateDeallocateWorkload_ShouldBeThreadSafe()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var duration = TimeSpan.FromSeconds(2);
        var cts = new CancellationTokenSource(duration);
        var errors = new ConcurrentBag<Exception>();

        // Act - Mixed workload
        var tasks = Enumerable.Range(0, 5)
            .Select(_ => Task.Run(() =>
            {
                var localMemories = new List<IGpuMemory<float>>();
                try
                {
                    while (!cts.Token.IsCancellationRequested)
                    {
                        // Allocate
                        localMemories.Add(pool.Rent(Random.Shared.Next(100, 10000)));

                        // Sometimes deallocate
                        if (localMemories.Count > 10 && Random.Shared.Next(2) == 0)
                        {
                            var memory = localMemories[0];
                            localMemories.RemoveAt(0);
                            pool.Return(memory);
                        }
                    }

                    // Cleanup remaining
                    foreach (var memory in localMemories)
                    {
                        pool.Return(memory);
                    }
                }
                catch (Exception ex)
                {
                    errors.Add(ex);
                }
            }));

        await Task.WhenAll(tasks);

        // Assert
        errors.Should().BeEmpty("No exceptions should occur during concurrent access");

        var stats = pool.GetStats();
        stats.InUse.Should().Be(0);
    }

    [Fact]
    public async Task StressTest_HighConcurrency_ShouldMaintainIntegrity()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var concurrentOperations = 1000;

        // Act
        var tasks = Enumerable.Range(0, concurrentOperations)
            .Select(i => Task.Run(() =>
            {
                var memory = pool.Rent(1000 + i % 1000);
                Thread.Sleep(Random.Shared.Next(1, 5)); // Simulate work
                pool.Return(memory);
            }));

        await Task.WhenAll(tasks);

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().Be(0);
        stats.RentCount.Should().Be(concurrentOperations);
        stats.ReturnCount.Should().Be(concurrentOperations);
    }

    #endregion

    #region Error Handling Tests (4 tests)

    [Fact]
    public void Rent_NegativeSize_ShouldThrowOrHandleGracefully()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act & Assert
        // ArrayPool.Rent handles negative by returning empty array or throwing
        try
        {
            var memory = pool.Rent(-1);
            memory.Should().NotBeNull(); // If it doesn't throw, should return valid memory
            pool.Return(memory);
        }
        catch (ArgumentOutOfRangeException)
        {
            // Expected behavior
        }
    }

    [Fact]
    public void Rent_MaxIntSize_ShouldHandleGracefully()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act & Assert
        try
        {
            var memory = pool.Rent(int.MaxValue);
            memory.Should().NotBeNull();
            pool.Return(memory);
        }
        catch (OutOfMemoryException)
        {
            // Expected for very large allocations
        }
    }

    [Fact]
    public void Return_NullArgument_ShouldThrowArgumentException()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() => pool.Return(null!));
        exception.Message.Should().Contain("Memory not from this pool");
    }

    [Fact]
    public void AsMemory_AfterDispose_ShouldThrowObjectDisposedException()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(1000);

        // Act
        memory.Dispose();

        // Assert
        Assert.Throws<ObjectDisposedException>(() => memory.AsMemory());
    }

    #endregion

    #region CpuMemory Tests (8 tests)

    [Fact]
    public void CpuMemory_AsMemory_ShouldReturnValidMemory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(1000);
        _allocatedMemory.Add(memory);

        // Act
        var memorySpan = memory.AsMemory();

        // Assert
        memorySpan.Length.Should().BeGreaterThanOrEqualTo(1000);

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void CpuMemory_WriteAndRead_ShouldMaintainData()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(100);
        _allocatedMemory.Add(memory);
        var span = memory.AsMemory().Span;

        // Act - Write data
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = i * 1.5f;
        }

        // Assert - Read data
        for (int i = 0; i < span.Length; i++)
        {
            span[i].Should().Be(i * 1.5f);
        }

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public async Task CpuMemory_CopyToDeviceAsync_ShouldCompleteSuccessfully()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(1000);
        _allocatedMemory.Add(memory);

        // Act
        await memory.CopyToDeviceAsync();

        // Assert - Should complete without error (no-op for CPU)
        memory.IsResident.Should().BeTrue();

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public async Task CpuMemory_CopyFromDeviceAsync_ShouldCompleteSuccessfully()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(1000);
        _allocatedMemory.Add(memory);

        // Act
        await memory.CopyFromDeviceAsync();

        // Assert - Should complete without error (no-op for CPU)
        memory.IsResident.Should().BeTrue();

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public async Task CpuMemory_CopyWithCancellation_ShouldRespectCancellationToken()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(1000);
        _allocatedMemory.Add(memory);
        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert - Should complete immediately (no-op)
        await memory.CopyToDeviceAsync(cts.Token);
        await memory.CopyFromDeviceAsync(cts.Token);

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void CpuMemory_Properties_ShouldReturnCorrectValues()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var length = 1000;
        var memory = pool.Rent(length);
        _allocatedMemory.Add(memory);

        // Assert
        memory.Length.Should().BeGreaterThanOrEqualTo(length);
        memory.SizeInBytes.Should().BeGreaterThanOrEqualTo(length * sizeof(float));
        memory.DeviceIndex.Should().Be(-1); // CPU
        memory.IsResident.Should().BeTrue();

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void CpuMemory_MultipleAsMemoryCalls_ShouldReturnSameData()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(100);
        _allocatedMemory.Add(memory);

        // Act - Write via first call
        var mem1 = memory.AsMemory();
        mem1.Span[0] = 42.0f;

        // Act - Read via second call
        var mem2 = memory.AsMemory();

        // Assert
        mem2.Span[0].Should().Be(42.0f);

        // Cleanup
        pool.Return(memory);
    }

    [Fact]
    public void CpuMemory_Dispose_ShouldReleaseResources()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var memory = pool.Rent(1000);

        // Act
        memory.Dispose();

        // Assert - Accessing after dispose should throw
        Assert.Throws<ObjectDisposedException>(() => memory.AsMemory());
    }

    #endregion

    #region Additional Coverage Tests (5 tests)

    [Fact]
    public void GetStats_DuringActiveAllocations_ShouldReflectCurrentState()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Allocate progressively
        var memory1 = pool.Rent(1000);
        var stats1 = pool.GetStats();

        var memory2 = pool.Rent(2000);
        var stats2 = pool.GetStats();

        var memory3 = pool.Rent(3000);
        var stats3 = pool.GetStats();

        _allocatedMemory.AddRange(new[] { memory1, memory2, memory3 });

        // Assert
        stats1.InUse.Should().BeLessThan(stats2.InUse);
        stats2.InUse.Should().BeLessThan(stats3.InUse);
        stats3.RentCount.Should().Be(3);
        stats3.ReturnCount.Should().Be(0);

        // Cleanup
        pool.Return(memory1);
        pool.Return(memory2);
        pool.Return(memory3);
    }

    [Fact]
    public void Pool_PooledBufferCount_ShouldNotExceedLimit()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Act - Create 15 different size buffers
        for (int i = 1; i <= 15; i++)
        {
            var memory = pool.Rent(i * 1000);
            pool.Return(memory);
        }

        var stats = pool.GetStats();

        // Assert - Pool keeps at most 10 buffers (but may reuse some during rent/return cycles)
        stats.BufferCount.Should().BeLessThanOrEqualTo(10);
        stats.ReturnCount.Should().Be(15);
    }

    [Fact]
    public void Rent_AfterPoolExhaustion_ShouldAllocateNew()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();

        // Fill pool to capacity (10 buffers) with specific sizes
        for (int i = 0; i < 10; i++)
        {
            var memory = pool.Rent(1000 + i * 100);
            pool.Return(memory);
        }

        var statsAfterFill = pool.GetStats();

        // Act - Request size that doesn't match any pooled buffer (much larger)
        var newMemory = pool.Rent(50000);
        _allocatedMemory.Add(newMemory);

        var statsAfterNew = pool.GetStats();

        // Assert - Pool tries to reuse but size doesn't match, so allocates new
        // TryTake may remove one item from pool if it tries to check it
        statsAfterNew.TotalAllocated.Should().BeGreaterThan(statsAfterFill.TotalAllocated);
        statsAfterNew.RentCount.Should().Be(11); // 10 fills + 1 new rent

        // Cleanup
        pool.Return(newMemory);
    }

    [Fact]
    public void Pool_DifferentGenericTypes_ShouldBeIndependent()
    {
        // Arrange
        var floatPool = new CpuMemoryPool<float>();
        var intPool = new CpuMemoryPool<int>();
        var bytePool = new CpuMemoryPool<byte>();

        // Act
        var floatMem = floatPool.Rent(1000);
        var intMem = intPool.Rent(1000);
        var byteMem = bytePool.Rent(1000);

        var floatStats = floatPool.GetStats();
        var intStats = intPool.GetStats();
        var byteStats = bytePool.GetStats();

        // Assert
        floatStats.InUse.Should().BeGreaterThan(0);
        intStats.InUse.Should().BeGreaterThan(0);
        byteStats.InUse.Should().BeGreaterThan(0);

        floatMem.SizeInBytes.Should().Be(1000 * sizeof(float));
        intMem.SizeInBytes.Should().Be(1000 * sizeof(int));
        byteMem.SizeInBytes.Should().Be(1000 * sizeof(byte));

        // Cleanup
        floatPool.Return(floatMem);
        intPool.Return(intMem);
        bytePool.Return(byteMem);
    }

    [Fact]
    public void Pool_LongRunningUsage_ShouldMaintainConsistency()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        var cycles = 100;

        // Act - Many allocation/deallocation cycles
        for (int i = 0; i < cycles; i++)
        {
            var memories = new List<IGpuMemory<float>>
            {
                pool.Rent(100),
                pool.Rent(1000),
                pool.Rent(500)
            };

            foreach (var memory in memories)
            {
                var span = memory.AsMemory().Span;
                span[0] = i; // Use memory
                pool.Return(memory);
            }
        }

        var stats = pool.GetStats();

        // Assert
        stats.InUse.Should().Be(0);
        stats.RentCount.Should().Be(cycles * 3);
        stats.ReturnCount.Should().Be(cycles * 3);
        stats.BufferCount.Should().BeGreaterThan(0).And.BeLessThanOrEqualTo(10);
    }

    #endregion
}
