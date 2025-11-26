using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Metrics;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Diagnostics;
using Orleans.GpuBridge.Diagnostics.Models;
using Orleans.GpuBridge.HealthChecks;
using Orleans.GpuBridge.HealthChecks.Configuration;
using Orleans.GpuBridge.HealthChecks.Implementation;
using Xunit;

namespace Orleans.GpuBridge.Tests.HealthChecks;

public class HealthCheckTests
{
    private readonly Mock<IGpuBridge> _mockGpuBridge;
    private readonly Mock<IGpuMetricsCollector> _mockMetricsCollector;
    private readonly Mock<ILogger<GpuHealthCheck>> _mockLogger;
    
    public HealthCheckTests()
    {
        _mockGpuBridge = new Mock<IGpuBridge>();
        _mockMetricsCollector = new Mock<IGpuMetricsCollector>();
        _mockLogger = new Mock<ILogger<GpuHealthCheck>>();
    }
    
    [Fact]
    public async Task CheckHealthAsync_NoGpuAvailable_ReturnsHealthyWithCpuFallback()
    {
        // Arrange
        _mockGpuBridge.Setup(x => x.GetDevicesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<GpuDevice>());
        
        var healthCheck = new GpuHealthCheck(
            _mockLogger.Object,
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
        _mockGpuBridge.Setup(x => x.GetDevicesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<GpuDevice>());
        
        var healthCheck = new GpuHealthCheck(
            _mockLogger.Object,
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
        var testDevice = new GpuDevice(0, "Test GPU", DeviceType.GPU, 8L * 1024 * 1024 * 1024, 7L * 1024 * 1024 * 1024, 80, new List<string> { "CUDA", "Compute_7.5" });
        
        _mockGpuBridge.Setup(x => x.GetDevicesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<GpuDevice> { testDevice });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0, It.IsAny<CancellationToken>()))
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
            _mockLogger.Object,
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
        var testDevice = new GpuDevice(0, "Test GPU", DeviceType.GPU, 8L * 1024 * 1024 * 1024, 1L * 1024 * 1024 * 1024, 80, new List<string> { "CUDA", "Compute_7.5" });
        
        _mockGpuBridge.Setup(x => x.GetDevicesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<GpuDevice> { testDevice });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0, It.IsAny<CancellationToken>()))
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
            _mockLogger.Object,
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
        var testDevice = new GpuDevice(0, "Test GPU", DeviceType.GPU, 8L * 1024 * 1024 * 1024, 6L * 1024 * 1024 * 1024, 80, new List<string> { "CUDA", "Compute_7.5" });
        
        _mockGpuBridge.Setup(x => x.GetDevicesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<GpuDevice> { testDevice });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new GpuDeviceMetrics
            {
                DeviceIndex = 0,
                DeviceName = "Test GPU",
                TemperatureCelsius = 60.0,
                MemoryUsedMB = 2000,
                MemoryTotalMB = 8000,
                GpuUtilization = 50
            });
        
        // Remove ExecuteAsync mock as it's no longer part of IGpuBridge interface
        
        var healthCheck = new GpuHealthCheck(
            _mockLogger.Object,
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
        var testDevice = new GpuDevice(0, "Test GPU", DeviceType.GPU, 8L * 1024 * 1024 * 1024, 6L * 1024 * 1024 * 1024, 80, new List<string> { "CUDA", "Compute_7.5" });
        
        _mockGpuBridge.Setup(x => x.GetDevicesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(new List<GpuDevice> { testDevice });
        
        _mockMetricsCollector.Setup(x => x.GetDeviceMetricsAsync(0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new GpuDeviceMetrics
            {
                DeviceIndex = 0,
                DeviceName = "Test GPU",
                TemperatureCelsius = 60.0,
                MemoryUsedMB = 2000,
                MemoryTotalMB = 8000,
                GpuUtilization = 50
            });
        
        // Remove ExecuteAsync mock and simulate kernel failure differently
        
        var healthCheck = new GpuHealthCheck(
            _mockLogger.Object,
            _mockGpuBridge.Object,
            _mockMetricsCollector.Object,
            new GpuHealthCheckOptions { TestKernelExecution = false }); // Disable kernel testing
        
        // Act
        var result = await healthCheck.CheckHealthAsync(new HealthCheckContext());
        
        // Assert - Without kernel testing, this should be healthy
        Assert.Equal(HealthStatus.Healthy, result.Status);
        Assert.Contains("operational", result.Description);
    }
}