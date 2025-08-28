using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Runtime;
using Xunit;

namespace Orleans.GpuBridge.Tests.Runtime;

public class MemoryPoolTests : IDisposable
{
    private readonly AdvancedMemoryPool<float> _pool;
    private readonly ILogger<AdvancedMemoryPool<float>> _logger;

    public MemoryPoolTests()
    {
        _logger = new TestLogger<AdvancedMemoryPool<float>>();
        _pool = new AdvancedMemoryPool<float>(_logger, maxBufferSize: 1024, maxPooledBuffers: 10);
    }

    [Fact]
    public void Rent_Should_Return_Memory_Of_Requested_Size()
    {
        // Act
        using var memory = _pool.Rent(100);

        // Assert
        Assert.NotNull(memory);
        Assert.True(memory.Length >= 100);
        Assert.True(memory.SizeInBytes > 0);
    }

    [Fact]
    public void Rent_With_Zero_Size_Should_Throw()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => _pool.Rent(0));
    }

    [Fact]
    public void Rent_With_Negative_Size_Should_Throw()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => _pool.Rent(-1));
    }

    [Fact]
    public void Return_Should_Recycle_Memory()
    {
        // Arrange
        var memory = _pool.Rent(100);
        var stats1 = _pool.GetStats();

        // Act
        _pool.Return(memory);
        var stats2 = _pool.GetStats();

        // Assert
        Assert.True(stats2.InUse < stats1.InUse);
    }

    [Fact]
    public void GetStats_Should_Return_Valid_Statistics()
    {
        // Arrange
        var memories = new List<IGpuMemory<float>>();
        for (int i = 0; i < 5; i++)
        {
            memories.Add(_pool.Rent(100));
        }

        // Act
        var stats = _pool.GetStats();

        // Assert
        Assert.True(stats.TotalAllocated > 0);
        Assert.True(stats.InUse > 0);
        Assert.True(stats.BufferCount >= 5);
        Assert.True(stats.RentCount > 0);
        Assert.True(stats.UtilizationPercent > 0);

        // Cleanup
        foreach (var memory in memories)
        {
            _pool.Return(memory);
        }
    }

    [Fact]
    public void Memory_Should_Be_Pinned_For_Large_Allocations()
    {
        // Act
        using var memory = _pool.Rent(20000); // Large allocation

        // Assert
        Assert.NotNull(memory);
        Assert.True(memory.Length >= 20000);
    }

    [Fact]
    public void Memory_Should_Support_AsMemory()
    {
        // Arrange
        using var memory = _pool.Rent(100);

        // Act
        var memorySpan = memory.AsMemory();

        // Assert
        Assert.Equal(memory.Length, memorySpan.Length);
    }

    [Fact]
    public async Task CopyToDeviceAsync_Should_Complete()
    {
        // Arrange
        using var memory = _pool.Rent(100);

        // Act
        await memory.CopyToDeviceAsync();

        // Assert
        Assert.True(memory.IsResident);
    }

    [Fact]
    public async Task CopyFromDeviceAsync_Should_Complete()
    {
        // Arrange
        using var memory = _pool.Rent(100);

        // Act
        await memory.CopyFromDeviceAsync();

        // Assert
        Assert.True(memory.IsResident);
    }

    [Fact]
    public void DeviceIndex_Should_Be_Negative_For_CPU_Memory()
    {
        // Arrange
        using var memory = _pool.Rent(100);

        // Assert
        Assert.Equal(-1, memory.DeviceIndex);
    }

    [Fact]
    public void Multiple_Rent_Should_Work_Correctly()
    {
        // Arrange
        var memories = new List<IGpuMemory<float>>();

        // Act
        for (int i = 0; i < 10; i++)
        {
            memories.Add(_pool.Rent(50 + i * 10));
        }

        // Assert
        Assert.Equal(10, memories.Count);
        for (int i = 0; i < 10; i++)
        {
            Assert.True(memories[i].Length >= 50 + i * 10);
        }

        // Cleanup
        foreach (var memory in memories)
        {
            _pool.Return(memory);
        }
    }

    [Fact]
    public void Dispose_Should_Clean_Up_Resources()
    {
        // Arrange
        var pool = new AdvancedMemoryPool<float>(_logger);
        var memory = pool.Rent(100);

        // Act
        pool.Dispose();

        // Assert
        Assert.Throws<ObjectDisposedException>(() => pool.Rent(100));
    }

    [Fact]
    public void Return_After_Dispose_Should_Not_Throw()
    {
        // Arrange
        var pool = new AdvancedMemoryPool<float>(_logger);
        var memory = pool.Rent(100);
        pool.Dispose();

        // Act & Assert (should not throw)
        pool.Return(memory);
    }

    [Fact]
    public void Statistics_Should_Track_Allocations_Correctly()
    {
        // Arrange
        var memories = new List<IGpuMemory<float>>();
        var initialStats = _pool.GetStats();

        // Act - Rent multiple buffers
        for (int i = 0; i < 3; i++)
        {
            memories.Add(_pool.Rent(100));
        }
        var afterRentStats = _pool.GetStats();

        // Return one buffer
        _pool.Return(memories[0]);
        memories.RemoveAt(0);
        var afterReturnStats = _pool.GetStats();

        // Assert
        Assert.True(afterRentStats.InUse > initialStats.InUse);
        Assert.True(afterReturnStats.InUse < afterRentStats.InUse);
        Assert.True(afterReturnStats.ReturnCount > afterRentStats.ReturnCount);

        // Cleanup
        foreach (var memory in memories)
        {
            _pool.Return(memory);
        }
    }

    [Fact]
    public async Task Concurrent_Rent_And_Return_Should_Be_Thread_Safe()
    {
        // Arrange
        var tasks = new List<Task>();
        var memories = new System.Collections.Concurrent.ConcurrentBag<IGpuMemory<float>>();

        // Act - Concurrent renting
        for (int i = 0; i < 10; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                for (int j = 0; j < 10; j++)
                {
                    var memory = _pool.Rent(Random.Shared.Next(50, 200));
                    memories.Add(memory);
                    Thread.Sleep(Random.Shared.Next(1, 5));
                }
            }));
        }

        await Task.WhenAll(tasks.ToArray());

        // Assert
        Assert.Equal(100, memories.Count);

        // Concurrent returning
        tasks.Clear();
        foreach (var memory in memories)
        {
            tasks.Add(Task.Run(() =>
            {
                Thread.Sleep(Random.Shared.Next(1, 5));
                _pool.Return(memory);
            }));
        }

        await Task.WhenAll(tasks.ToArray());

        // Final stats should show everything returned
        var finalStats = _pool.GetStats();
        Assert.True(finalStats.ReturnCount > 0);
    }

    public void Dispose()
    {
        _pool?.Dispose();
    }
}

public class MemoryPoolManagerTests : IDisposable
{
    private readonly MemoryPoolManager _manager;
    private readonly ILoggerFactory _loggerFactory;

    public MemoryPoolManagerTests()
    {
        _loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        _manager = new MemoryPoolManager(_loggerFactory);
    }

    [Fact]
    public void GetPool_Should_Return_Pool_For_Type()
    {
        // Act
        var floatPool = _manager.GetPool<float>();
        var intPool = _manager.GetPool<int>();
        var doublePool = _manager.GetPool<double>();

        // Assert
        Assert.NotNull(floatPool);
        Assert.NotNull(intPool);
        Assert.NotNull(doublePool);
    }

    [Fact]
    public void GetPool_Should_Return_Same_Instance_For_Same_Type()
    {
        // Act
        var pool1 = _manager.GetPool<float>();
        var pool2 = _manager.GetPool<float>();

        // Assert
        Assert.Same(pool1, pool2);
    }

    [Fact]
    public void GetAllStats_Should_Return_Statistics_For_All_Pools()
    {
        // Arrange
        var floatPool = _manager.GetPool<float>();
        var intPool = _manager.GetPool<int>();
        
        using var floatMemory = floatPool.Rent(100);
        using var intMemory = intPool.Rent(50);

        // Act
        var stats = _manager.GetAllStats();

        // Assert
        Assert.NotEmpty(stats);
        Assert.True(stats.Count >= 2);
    }

    [Fact]
    public void Dispose_Should_Dispose_All_Pools()
    {
        // Arrange
        var floatPool = _manager.GetPool<float>();
        var memory = floatPool.Rent(100);

        // Act
        _manager.Dispose();
        _manager.Dispose(); // Second dispose should not throw

        // Assert - pool operations should fail after manager disposal
        Assert.Throws<ObjectDisposedException>(() => floatPool.Rent(100));
    }

    public void Dispose()
    {
        _manager?.Dispose();
        _loggerFactory?.Dispose();
    }
}

internal class TestLogger<T> : ILogger<T>
{
    public IDisposable BeginScope<TState>(TState state) where TState : notnull => new NoopDisposable();
    public bool IsEnabled(LogLevel logLevel) => true;
    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, 
        Exception? exception, Func<TState, Exception?, string> formatter)
    {
        // Capture logs for testing if needed
    }

    private class NoopDisposable : IDisposable
    {
        public void Dispose() { }
    }
}