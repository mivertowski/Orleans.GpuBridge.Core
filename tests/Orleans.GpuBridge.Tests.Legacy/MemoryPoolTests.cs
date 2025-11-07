using FluentAssertions;
using Orleans.GpuBridge.Runtime;
using Xunit;

namespace Orleans.GpuBridge.Tests;

public class MemoryPoolTests
{
    [Fact]
    public void Should_Rent_And_Return_Memory()
    {
        // Arrange
        var pool = new CpuMemoryPool<float>();
        
        // Act
        var memory = pool.Rent(1024);
        var stats1 = pool.GetStats();
        
        pool.Return(memory);
        var stats2 = pool.GetStats();
        
        // Assert
        memory.Should().NotBeNull();
        memory.Length.Should().BeGreaterThanOrEqualTo(1024);
        
        stats1.InUse.Should().BeGreaterThan(0);
        stats1.RentCount.Should().Be(1);
        
        stats2.InUse.Should().Be(0);
        stats2.ReturnCount.Should().Be(1);
    }
    
    [Fact]
    public void Should_Reuse_Pooled_Buffers()
    {
        // Arrange
        var pool = new CpuMemoryPool<int>();
        
        // Act
        var memory1 = pool.Rent(512);
        pool.Return(memory1);
        
        var memory2 = pool.Rent(512);
        pool.Return(memory2);
        
        var stats = pool.GetStats();
        
        // Assert
        stats.BufferCount.Should().BeGreaterThan(0);
        stats.RentCount.Should().Be(2);
        stats.ReturnCount.Should().Be(2);
    }
    
    [Fact]
    public async Task Memory_Should_Support_Copy_Operations()
    {
        // Arrange
        var pool = new CpuMemoryPool<byte>();
        var memory = pool.Rent(256);
        
        // Act
        var data = memory.AsMemory();
        data.Span.Fill(42);
        
        await memory.CopyToDeviceAsync();
        await memory.CopyFromDeviceAsync();
        
        // Assert
        memory.IsResident.Should().BeTrue();
        memory.DeviceIndex.Should().Be(-1); // CPU
        data.Span[0].Should().Be(42);
    }
    
    [Fact]
    public void Should_Track_Memory_Statistics()
    {
        // Arrange
        var pool = new CpuMemoryPool<double>();
        
        // Act
        var memories = new List<Orleans.GpuBridge.Abstractions.Memory.IGpuMemory<double>>();
        for (int i = 0; i < 5; i++)
        {
            memories.Add(pool.Rent(100));
        }
        
        var statsInUse = pool.GetStats();
        
        foreach (var mem in memories)
        {
            pool.Return(mem);
        }
        
        var statsReturned = pool.GetStats();
        
        // Assert
        statsInUse.InUse.Should().BeGreaterThan(0);
        statsInUse.UtilizationPercent.Should().BeGreaterThan(0);
        
        statsReturned.InUse.Should().Be(0);
        statsReturned.Available.Should().BeGreaterThan(0);
    }
}