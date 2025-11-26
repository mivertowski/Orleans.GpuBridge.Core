using Xunit;
using FluentAssertions;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Tests.SimpleTests;

public class BasicTests
{
    [Fact]
    public void SimpleTest_ShouldPass()
    {
        // Arrange
        var expected = 2 + 2;
        
        // Act
        var actual = 4;
        
        // Assert
        actual.Should().Be(expected);
    }
    
    [Fact]
    public void KernelId_ShouldCreateCorrectly()
    {
        // Arrange & Act
        var kernelId = new KernelId("test-kernel");
        
        // Assert
        kernelId.Value.Should().Be("test-kernel");
        kernelId.ToString().Should().Be("test-kernel");
    }
    
    [Fact]
    public void GpuBridgeOptions_ShouldHaveDefaultValues()
    {
        // Arrange & Act
        var options = new GpuBridgeOptions();
        
        // Assert
        options.PreferGpu.Should().BeTrue();
        options.EnableMetrics.Should().BeTrue();
        options.BatchSize.Should().Be(1024);
    }
}