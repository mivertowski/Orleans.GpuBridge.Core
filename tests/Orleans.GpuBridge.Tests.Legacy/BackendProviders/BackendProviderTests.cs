using System;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Tests.TestingFramework;
using DeviceType = Orleans.GpuBridge.Abstractions.Enums.DeviceType;
using Xunit;

namespace Orleans.GpuBridge.Tests.BackendProviders;

/// <summary>
/// Tests for backend provider functionality
/// </summary>
public class BackendProviderTests
{
    private readonly IServiceProvider _serviceProvider;

    public BackendProviderTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddGpuBridge();
        _serviceProvider = services.BuildServiceProvider();
    }

    [Fact]
    public async Task TestGpuProvider_Should_Be_Available()
    {
        // Arrange
        var testProvider = new TestGpuProvider();
        
        // Act
        var isAvailable = await testProvider.IsAvailableAsync();
        
        // Assert
        Assert.True(isAvailable);
        Assert.Equal("TestGpu", testProvider.ProviderId);
        Assert.Equal("Test GPU Provider", testProvider.DisplayName);
    }

    [Fact]
    public async Task TestCpuProvider_Should_Be_Available()
    {
        // Arrange
        var testProvider = new TestCpuProvider();
        
        // Act
        var isAvailable = await testProvider.IsAvailableAsync();
        
        // Assert
        Assert.True(isAvailable);
        Assert.Equal("TestCpu", testProvider.ProviderId);
        Assert.Equal("Test CPU Provider", testProvider.DisplayName);
    }

    [Fact]
    public async Task Provider_GetMetrics_Should_Return_Valid_Data()
    {
        // Arrange
        var testProvider = new TestGpuProvider();
        
        // Act
        var metrics = await testProvider.GetMetricsAsync();
        
        // Assert
        Assert.NotNull(metrics);
        Assert.Contains("MemoryUsage", metrics.Keys);
        Assert.Contains("GpuUtilization", metrics.Keys);
        Assert.Contains("Temperature", metrics.Keys);
    }

    [Fact]
    public async Task Provider_HealthCheck_Should_Return_Healthy()
    {
        // Arrange
        var testProvider = new TestGpuProvider();
        
        // Act
        var healthResult = await testProvider.CheckHealthAsync();
        
        // Assert
        Assert.NotNull(healthResult);
        Assert.True(healthResult.IsHealthy);
    }

    [Fact]
    public void Provider_GetDeviceManager_Should_Return_Valid_Manager()
    {
        // Arrange
        var testProvider = new TestGpuProvider();
        
        // Act
        var deviceManager = testProvider.GetDeviceManager();
        
        // Assert
        Assert.NotNull(deviceManager);
    }

    [Fact]
    public void Provider_GetKernelCompiler_Should_Return_Valid_Compiler()
    {
        // Arrange
        var testProvider = new TestGpuProvider();
        
        // Act
        var compiler = testProvider.GetKernelCompiler();
        
        // Assert
        Assert.NotNull(compiler);
    }

    [Fact]
    public void Provider_GetMemoryAllocator_Should_Return_Valid_Allocator()
    {
        // Arrange
        var testProvider = new TestGpuProvider();
        
        // Act
        var allocator = testProvider.GetMemoryAllocator();
        
        // Assert
        Assert.NotNull(allocator);
    }

    [Fact]
    public void Provider_GetKernelExecutor_Should_Return_Valid_Executor()
    {
        // Arrange
        var testProvider = new TestGpuProvider();
        
        // Act
        var executor = testProvider.GetKernelExecutor();
        
        // Assert
        Assert.NotNull(executor);
    }
}

/// <summary>
/// Tests specifically for device type functionality
/// </summary>
public class DeviceTypeTests
{
    [Theory]
    [InlineData(DeviceType.CPU)]
    [InlineData(DeviceType.GPU)]
    [InlineData(DeviceType.CUDA)]
    [InlineData(DeviceType.OpenCL)]
    public void DeviceType_Should_Have_Valid_Values(DeviceType deviceType)
    {
        // Assert
        Assert.True(Enum.IsDefined(typeof(DeviceType), deviceType));
    }
}

/// <summary>
/// Tests for buffer usage flags
/// </summary>
public class BufferUsageTests
{
    [Theory]
    [InlineData(BufferUsage.ReadOnly)]
    [InlineData(BufferUsage.WriteOnly)]
    [InlineData(BufferUsage.ReadWrite)]
    [InlineData(BufferUsage.Persistent)]
    [InlineData(BufferUsage.Streaming)]
    [InlineData(BufferUsage.UnifiedMemory)]
    public void BufferUsage_Should_Have_Valid_Values(BufferUsage usage)
    {
        // Assert
        Assert.True(Enum.IsDefined(typeof(BufferUsage), usage));
    }

    [Fact]
    public void BufferUsage_ReadWrite_Should_Combine_Read_And_Write()
    {
        // Act & Assert
        Assert.Equal(BufferUsage.ReadOnly | BufferUsage.WriteOnly, BufferUsage.ReadWrite);
    }
}