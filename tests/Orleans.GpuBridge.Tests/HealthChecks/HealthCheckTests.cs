using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Diagnostics;
using Orleans.GpuBridge.Diagnostics.Abstractions;
using Orleans.GpuBridge.HealthChecks;
using Orleans.GpuBridge.HealthChecks.Implementation;
using Xunit;

namespace Orleans.GpuBridge.Tests.HealthChecks;

public class HealthCheckTests
{
    private readonly Mock<IGpuBridge> _mockGpuBridge;
    private readonly Mock<IGpuMetricsCollector> _mockMetricsCollector;
    private readonly ILogger<GpuHealthCheck> _logger;
    
    public HealthCheckTests()
    {
        _mockGpuBridge = new Mock<IGpuBridge>();
        _mockMetricsCollector = new Mock<IGpuMetricsCollector>();
        _logger = new TestLogger<GpuHealthCheck>();
    }
    
    [Fact]
    public async Task CheckHealthAsync_NoGpuAvailable_ReturnsHealthyWithCpuFallback()
    {
        // Arrange
        _mockGpuBridge.Setup(x => x.GetAvailableDevicesAsync())
            .ReturnsAsync(new List<IGpuDevice>());
        
        var healthCheck = new GpuHealthCheck(
            _logger,
            _mockGpuBridge.Object,
            _mockMetricsCollector.Object,
            new GpuHealthCheckOptions { RequireGpu = false });
        
        // Act
        var result = await healthCheck.CheckHealthAsync(new HealthCheckContext());
        
        // Assert
        Assert.Equal(HealthStatus.Degraded, result.Status);
        Assert.Contains("No GPU devices available", result.Description);
        Assert.Contains("device_count", result.Data.Keys);
        Assert.Equal(0, result.Data["device_count"]);
    }
    
    [Fact]
    public async Task CheckHealthAsync_NoGpuRequired_ReturnsUnhealthy()
    {
        // Arrange
        _mockGpuBridge.Setup(x => x.GetAvailableDevicesAsync())
            .ReturnsAsync(new List<IGpuDevice>());
        
        var healthCheck = new GpuHealthCheck(
            _logger,
            _mockGpuBridge.Object,
            _mockMetricsCollector.Object,
            new GpuHealthCheckOptions { RequireGpu = true });
        
        // Act
        var result = await healthCheck.CheckHealthAsync(new HealthCheckContext());
        
        // Assert
        Assert.Equal(HealthStatus.Unhealthy, result.Status);
        Assert.Contains("No GPU devices available", result.Description);
    }
    
    [Fact]
    public async Task CheckHealthAsync_HighTemperature_ReturnsUnhealthy()
    {
        // Arrange
        var mockDevice = new Mock<IGpuDevice>();
        mockDevice.Setup(x => x.Index).Returns(0);
        mockDevice.Setup(x => x.Name).Returns("Test GPU");
        
        _mockGpuBridge.Setup(x => x.GetAvailableDevicesAsync())
            .ReturnsAsync(new List<IGpuDevice> { mockDevice.Object });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0))
            .ReturnsAsync(new GpuDeviceMetrics
            {
                DeviceIndex = 0,
                DeviceName = "Test GPU",
                TemperatureCelsius = 90.0, // Above max threshold
                MemoryUsedMB = 1000,
                MemoryTotalMB = 8000,
                GpuUtilization = 50
            });
        
        var healthCheck = new GpuHealthCheck(
            _logger,
            _mockGpuBridge.Object,
            _mockMetricsCollector.Object,
            new GpuHealthCheckOptions { MaxTemperatureCelsius = 85.0 });
        
        // Act
        var result = await healthCheck.CheckHealthAsync(new HealthCheckContext());
        
        // Assert
        Assert.Equal(HealthStatus.Unhealthy, result.Status);
        Assert.Contains("temperature too high", result.Description);
    }
    
    [Fact]
    public async Task CheckHealthAsync_HighMemoryUsage_ReturnsDegraded()
    {
        // Arrange
        var mockDevice = new Mock<IGpuDevice>();
        mockDevice.Setup(x => x.Index).Returns(0);
        
        _mockGpuBridge.Setup(x => x.GetAvailableDevicesAsync())
            .ReturnsAsync(new List<IGpuDevice> { mockDevice.Object });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0))
            .ReturnsAsync(new GpuDeviceMetrics
            {
                DeviceIndex = 0,
                DeviceName = "Test GPU",
                TemperatureCelsius = 70.0,
                MemoryUsedMB = 7000,
                MemoryTotalMB = 8000, // 87.5% usage
                GpuUtilization = 50
            });
        
        var healthCheck = new GpuHealthCheck(
            _logger,
            _mockGpuBridge.Object,
            _mockMetricsCollector.Object,
            new GpuHealthCheckOptions 
            { 
                WarnMemoryUsagePercent = 80.0,
                MaxMemoryUsagePercent = 95.0
            });
        
        // Act
        var result = await healthCheck.CheckHealthAsync(new HealthCheckContext());
        
        // Assert
        Assert.Equal(HealthStatus.Degraded, result.Status);
        Assert.Contains("memory pressure", result.Description);
    }
    
    [Fact]
    public async Task CheckHealthAsync_AllHealthy_ReturnsHealthy()
    {
        // Arrange
        var mockDevice = new Mock<IGpuDevice>();
        mockDevice.Setup(x => x.Index).Returns(0);
        
        _mockGpuBridge.Setup(x => x.GetAvailableDevicesAsync())
            .ReturnsAsync(new List<IGpuDevice> { mockDevice.Object });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0))
            .ReturnsAsync(new GpuDeviceMetrics
            {
                DeviceIndex = 0,
                DeviceName = "Test GPU",
                TemperatureCelsius = 60.0,
                MemoryUsedMB = 2000,
                MemoryTotalMB = 8000,
                GpuUtilization = 50
            });
        
        _mockGpuBridge.Setup(x => x.ExecuteAsync<float[], float>(
                It.IsAny<string>(), 
                It.IsAny<float[]>(), 
                It.IsAny<GpuExecutionHints>()))
            .ReturnsAsync(10.0f); // Sum of test array
        
        var healthCheck = new GpuHealthCheck(
            _logger,
            _mockGpuBridge.Object,
            _mockMetricsCollector.Object);
        
        // Act
        var result = await healthCheck.CheckHealthAsync(new HealthCheckContext());
        
        // Assert
        Assert.Equal(HealthStatus.Healthy, result.Status);
        Assert.Contains("operational", result.Description);
    }
    
    [Fact]
    public async Task CheckHealthAsync_KernelTestFails_ReturnsDegraded()
    {
        // Arrange
        var mockDevice = new Mock<IGpuDevice>();
        mockDevice.Setup(x => x.Index).Returns(0);
        
        _mockGpuBridge.Setup(x => x.GetAvailableDevicesAsync())
            .ReturnsAsync(new List<IGpuDevice> { mockDevice.Object });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0))
            .ReturnsAsync(new GpuDeviceMetrics
            {
                DeviceIndex = 0,
                DeviceName = "Test GPU",
                TemperatureCelsius = 60.0,
                MemoryUsedMB = 2000,
                MemoryTotalMB = 8000,
                GpuUtilization = 50
            });
        
        _mockGpuBridge.Setup(x => x.ExecuteAsync<float[], float>(
                It.IsAny<string>(), 
                It.IsAny<float[]>(), 
                It.IsAny<GpuExecutionHints>()))
            .ThrowsAsync(new Exception("Kernel execution failed"));
        
        var healthCheck = new GpuHealthCheck(
            _logger,
            _mockGpuBridge.Object,
            _mockMetricsCollector.Object,
            new GpuHealthCheckOptions { TestKernelExecution = true });
        
        // Act
        var result = await healthCheck.CheckHealthAsync(new HealthCheckContext());
        
        // Assert
        Assert.Equal(HealthStatus.Degraded, result.Status);
        Assert.Contains("Kernel test error", result.Description);
    }
}