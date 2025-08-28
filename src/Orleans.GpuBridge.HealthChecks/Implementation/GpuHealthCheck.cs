using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Diagnostics;
using Orleans.GpuBridge.HealthChecks.Configuration;

namespace Orleans.GpuBridge.HealthChecks.Implementation;

/// <summary>
/// Health check implementation for monitoring GPU system health, performance, and functionality.
/// This service evaluates GPU hardware metrics, device availability, memory utilization,
/// and optionally performs functional testing to ensure comprehensive system health monitoring.
/// </summary>
/// <remarks>
/// The health check performs multi-dimensional assessment:
/// 
/// **Device Availability**: Verifies GPU devices are present and accessible
/// **Hardware Metrics**: Monitors temperature, power consumption, and utilization
/// **Memory Health**: Tracks memory usage, pressure, and allocation patterns  
/// **Functional Testing**: Optionally executes test kernels to verify compute functionality
/// **Performance Monitoring**: Evaluates utilization patterns and efficiency
/// 
/// Health status determination:
/// - **Healthy**: All metrics within acceptable ranges, devices operational
/// - **Degraded**: Warning thresholds exceeded, performance issues, or minor problems
/// - **Unhealthy**: Critical thresholds exceeded, device failures, or system errors
/// </remarks>
public class GpuHealthCheck : IHealthCheck
{
    private readonly ILogger<GpuHealthCheck> _logger;
    private readonly IGpuBridge _gpuBridge;
    private readonly IGpuMetricsCollector _metricsCollector;
    private readonly GpuHealthCheckOptions _options;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="GpuHealthCheck"/> class.
    /// </summary>
    /// <param name="logger">Logger for recording health check operations and results.</param>
    /// <param name="gpuBridge">GPU bridge service for device access and kernel execution.</param>
    /// <param name="metricsCollector">Metrics collector for retrieving GPU hardware statistics.</param>
    /// <param name="options">Configuration options controlling health check behavior and thresholds.</param>
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
    
    /// <summary>
    /// Performs a comprehensive health assessment of the GPU system.
    /// This method evaluates device availability, hardware metrics, memory utilization,
    /// and optionally functional testing to determine overall system health.
    /// </summary>
    /// <param name="context">Health check context containing registration information.</param>
    /// <param name="cancellationToken">Cancellation token for the health check operation.</param>
    /// <returns>
    /// A <see cref="HealthCheckResult"/> indicating system health status with detailed diagnostic data.
    /// </returns>
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var data = new Dictionary<string, object>();
            var unhealthyReasons = new List<string>();
            var degradedReasons = new List<string>();
            
            // Phase 1: Check GPU device availability
            var devices = await _gpuBridge.GetDevicesAsync();
            data["device_count"] = devices.Count;
            
            if (devices.Count == 0)
            {
                if (_options.RequireGpu)
                {
                    return HealthCheckResult.Unhealthy(
                        "No GPU devices available and GPU is required",
                        data: data);
                }
                else
                {
                    degradedReasons.Add("No GPU devices available (CPU fallback active)");
                }
            }
            
            // Phase 2: Evaluate individual device health metrics
            var healthyDevices = 0;
            var totalMemoryMB = 0L;
            var usedMemoryMB = 0L;
            
            foreach (var device in devices)
            {
                try
                {
                    var metrics = await _metricsCollector.GetDeviceMetricsAsync(device.Index);
                    
                    // Record device information in health check data
                    data[$"device_{device.Index}_name"] = metrics.DeviceName;
                    data[$"device_{device.Index}_utilization"] = $"{metrics.GpuUtilization:F1}%";
                    data[$"device_{device.Index}_memory"] = $"{metrics.MemoryUsedMB}/{metrics.MemoryTotalMB} MB";
                    data[$"device_{device.Index}_temperature"] = $"{metrics.TemperatureCelsius:F1}°C";
                    data[$"device_{device.Index}_power"] = $"{metrics.PowerUsageWatts:F1}W";
                    
                    totalMemoryMB += metrics.MemoryTotalMB;
                    usedMemoryMB += metrics.MemoryUsedMB;
                    
                    // Evaluate temperature thresholds
                    if (metrics.TemperatureCelsius > _options.MaxTemperatureCelsius)
                    {
                        unhealthyReasons.Add($"Device {device.Index} temperature too high: {metrics.TemperatureCelsius:F1}°C (max: {_options.MaxTemperatureCelsius:F1}°C)");
                    }
                    else if (metrics.TemperatureCelsius > _options.WarnTemperatureCelsius)
                    {
                        degradedReasons.Add($"Device {device.Index} temperature elevated: {metrics.TemperatureCelsius:F1}°C (warn: {_options.WarnTemperatureCelsius:F1}°C)");
                    }
                    
                    // Evaluate memory utilization thresholds
                    if (metrics.MemoryUsagePercent > _options.MaxMemoryUsagePercent)
                    {
                        unhealthyReasons.Add($"Device {device.Index} memory exhausted: {metrics.MemoryUsagePercent:F1}% (max: {_options.MaxMemoryUsagePercent:F1}%)");
                    }
                    else if (metrics.MemoryUsagePercent > _options.WarnMemoryUsagePercent)
                    {
                        degradedReasons.Add($"Device {device.Index} memory pressure: {metrics.MemoryUsagePercent:F1}% (warn: {_options.WarnMemoryUsagePercent:F1}%)");
                    }
                    
                    // Evaluate utilization efficiency
                    if (metrics.GpuUtilization < _options.MinUtilizationPercent)
                    {
                        degradedReasons.Add($"Device {device.Index} underutilized: {metrics.GpuUtilization:F1}% (min: {_options.MinUtilizationPercent:F1}%)");
                    }
                    
                    healthyDevices++;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to retrieve metrics for GPU device {DeviceIndex}", device.Index);
                    degradedReasons.Add($"Device {device.Index} metrics unavailable: {ex.Message}");
                }
            }
            
            // Record aggregate system information
            data["healthy_devices"] = healthyDevices;
            data["total_memory_gb"] = Math.Round(totalMemoryMB / 1024.0, 2);
            data["used_memory_gb"] = Math.Round(usedMemoryMB / 1024.0, 2);
            
            // Phase 3: Optional functional testing
            if (_options.TestKernelExecution && devices.Count > 0)
            {
                try
                {
                    var testResult = await TestKernelExecutionAsync(cancellationToken);
                    data["kernel_test"] = testResult ? "passed" : "failed";
                    
                    if (!testResult)
                    {
                        degradedReasons.Add("Kernel execution test failed - compute functionality may be impaired");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Kernel execution test encountered an error");
                    degradedReasons.Add($"Kernel test error: {ex.Message}");
                }
            }
            
            // Phase 4: Determine overall health status based on collected issues
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
                $"{healthyDevices} GPU device(s) operational, {totalMemoryMB / 1024.0:F1} GB total memory",
                data: data);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "GPU health check failed with unexpected error");
            
            return HealthCheckResult.Unhealthy(
                "GPU health check encountered an error",
                exception: ex,
                data: new Dictionary<string, object>
                {
                    ["error"] = ex.Message,
                    ["error_type"] = ex.GetType().Name
                });
        }
    }
    
    /// <summary>
    /// Performs functional testing by executing a simple kernel to verify GPU compute capability.
    /// This test validates that the GPU can successfully execute kernels and return correct results.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token for the test operation.</param>
    /// <returns>
    /// <c>true</c> if the kernel execution test passes and produces expected results;
    /// <c>false</c> if the test fails or produces incorrect results.
    /// </returns>
    /// <remarks>
    /// The test performs a simple sum operation on a small array of floating-point values
    /// to verify basic GPU compute functionality. The test is designed to be:
    /// - Lightweight: Minimal resource usage and execution time
    /// - Deterministic: Produces predictable, verifiable results
    /// - Representative: Uses typical GPU compute patterns (data parallel processing)
    /// </remarks>
    private async Task<bool> TestKernelExecutionAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Create simple test data for kernel execution verification
            var testData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var expectedSum = testData.Sum(); // Expected result: 10.0f
            
            // Attempt to get and execute a test kernel
            var kernel = await _gpuBridge.GetKernelAsync<float[], float>(
                new KernelId("test/sum"));
            
            // Submit test data for processing
            var handle = await kernel.SubmitBatchAsync(new[] { testData });
            
            // Retrieve and validate results
            var results = kernel.ReadResultsAsync(handle);
            var result = 0f;
            await foreach (var r in results)
            {
                result = r;
                break; // Get first result only
            }
            
            // Verify that the computed result matches expected value
            var tolerance = 0.001f; // Allow for small floating-point precision differences
            var resultValid = Math.Abs(result - expectedSum) < tolerance;
            
            if (!resultValid)
            {
                _logger.LogWarning(
                    "Kernel execution test failed: expected {Expected}, got {Actual}",
                    expectedSum, result);
            }
            
            return resultValid;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Kernel execution test failed with exception");
            return false;
        }
    }
}