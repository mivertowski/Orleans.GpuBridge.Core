using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Diagnostics;

namespace Orleans.GpuBridge.HealthChecks;

public class GpuHealthCheck : IHealthCheck
{
    private readonly ILogger<GpuHealthCheck> _logger;
    private readonly IGpuBridge _gpuBridge;
    private readonly IGpuMetricsCollector _metricsCollector;
    private readonly GpuHealthCheckOptions _options;
    
    public GpuHealthCheck(
        ILogger<GpuHealthCheck> logger,
        IGpuBridge gpuBridge,
        IGpuMetricsCollector metricsCollector,
        GpuHealthCheckOptions? options = null)
    {
        _logger = logger;
        _gpuBridge = gpuBridge;
        _metricsCollector = metricsCollector;
        _options = options ?? new GpuHealthCheckOptions();
    }
    
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var data = new Dictionary<string, object>();
            var unhealthyReasons = new List<string>();
            var degradedReasons = new List<string>();
            
            // Check GPU availability
            var devices = await _gpuBridge.GetDevicesAsync();
            data["device_count"] = devices.Count;
            
            if (devices.Count == 0)
            {
                if (_options.RequireGpu)
                {
                    return HealthCheckResult.Unhealthy(
                        "No GPU devices available",
                        data: data);
                }
                else
                {
                    degradedReasons.Add("No GPU devices available (CPU fallback active)");
                }
            }
            
            // Check each device
            var healthyDevices = 0;
            var totalMemoryMB = 0L;
            var usedMemoryMB = 0L;
            
            foreach (var device in devices)
            {
                try
                {
                    var metrics = await _metricsCollector.GetDeviceMetricsAsync(device.Index);
                    
                    data[$"device_{device.Index}_name"] = metrics.DeviceName;
                    data[$"device_{device.Index}_utilization"] = $"{metrics.GpuUtilization:F1}%";
                    data[$"device_{device.Index}_memory"] = $"{metrics.MemoryUsedMB}/{metrics.MemoryTotalMB} MB";
                    data[$"device_{device.Index}_temperature"] = $"{metrics.TemperatureCelsius:F1}°C";
                    data[$"device_{device.Index}_power"] = $"{metrics.PowerUsageWatts:F1}W";
                    
                    totalMemoryMB += metrics.MemoryTotalMB;
                    usedMemoryMB += metrics.MemoryUsedMB;
                    
                    // Check device health thresholds
                    if (metrics.TemperatureCelsius > _options.MaxTemperatureCelsius)
                    {
                        unhealthyReasons.Add($"Device {device.Index} temperature too high: {metrics.TemperatureCelsius:F1}°C");
                    }
                    else if (metrics.TemperatureCelsius > _options.WarnTemperatureCelsius)
                    {
                        degradedReasons.Add($"Device {device.Index} temperature elevated: {metrics.TemperatureCelsius:F1}°C");
                    }
                    
                    if (metrics.MemoryUsagePercent > _options.MaxMemoryUsagePercent)
                    {
                        unhealthyReasons.Add($"Device {device.Index} memory exhausted: {metrics.MemoryUsagePercent:F1}%");
                    }
                    else if (metrics.MemoryUsagePercent > _options.WarnMemoryUsagePercent)
                    {
                        degradedReasons.Add($"Device {device.Index} memory pressure: {metrics.MemoryUsagePercent:F1}%");
                    }
                    
                    if (metrics.GpuUtilization < _options.MinUtilizationPercent)
                    {
                        degradedReasons.Add($"Device {device.Index} underutilized: {metrics.GpuUtilization:F1}%");
                    }
                    
                    healthyDevices++;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to get metrics for device {DeviceIndex}", device.Index);
                    degradedReasons.Add($"Device {device.Index} metrics unavailable");
                }
            }
            
            data["healthy_devices"] = healthyDevices;
            data["total_memory_gb"] = totalMemoryMB / 1024.0;
            data["used_memory_gb"] = usedMemoryMB / 1024.0;
            
            // Test kernel execution if configured
            if (_options.TestKernelExecution && devices.Count > 0)
            {
                try
                {
                    var testResult = await TestKernelExecutionAsync(cancellationToken);
                    data["kernel_test"] = testResult ? "passed" : "failed";
                    
                    if (!testResult)
                    {
                        degradedReasons.Add("Kernel execution test failed");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Kernel execution test failed");
                    degradedReasons.Add($"Kernel test error: {ex.Message}");
                }
            }
            
            // Determine overall health status
            if (unhealthyReasons.Count > 0)
            {
                return HealthCheckResult.Unhealthy(
                    string.Join("; ", unhealthyReasons),
                    data: data);
            }
            
            if (degradedReasons.Count > 0)
            {
                return HealthCheckResult.Degraded(
                    string.Join("; ", degradedReasons),
                    data: data);
            }
            
            return HealthCheckResult.Healthy(
                $"{healthyDevices} GPU device(s) operational",
                data: data);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "GPU health check failed");
            
            return HealthCheckResult.Unhealthy(
                "GPU health check error",
                exception: ex,
                data: new Dictionary<string, object>
                {
                    ["error"] = ex.Message
                });
        }
    }
    
    private async Task<bool> TestKernelExecutionAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Simple test kernel execution
            var testData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var kernel = await _gpuBridge.GetKernelAsync<float[], float>(
                new KernelId("test/sum"));
            var handle = await kernel.SubmitBatchAsync(new[] { testData });
            var results = kernel.ReadResultsAsync(handle);
            var result = 0f;
            await foreach (var r in results)
            {
                result = r;
                break; // Get first result
            }
            
            // Verify result
            var expected = testData.Sum();
            return Math.Abs(result - expected) < 0.001f;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Test kernel execution failed");
            return false;
        }
    }
}

public class GpuHealthCheckOptions
{
    public bool RequireGpu { get; set; } = false;
    public bool TestKernelExecution { get; set; } = true;
    public double MaxTemperatureCelsius { get; set; } = 85.0;
    public double WarnTemperatureCelsius { get; set; } = 75.0;
    public double MaxMemoryUsagePercent { get; set; } = 95.0;
    public double WarnMemoryUsagePercent { get; set; } = 80.0;
    public double MinUtilizationPercent { get; set; } = 5.0;
}