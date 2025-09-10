using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using DeviceStatus = Orleans.GpuBridge.Abstractions.Enums.DeviceStatus;
using DeviceHealthInfo = Orleans.GpuBridge.Abstractions.Models.DeviceHealthInfo;
using DeviceLoadInfo = Orleans.GpuBridge.Abstractions.Models.DeviceLoadInfo;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Production-grade helper methods for DeviceBroker
/// </summary>
public sealed partial class DeviceBroker
{
    /// <summary>
    /// Asynchronous health monitoring for production environments
    /// </summary>
    private async Task MonitorDeviceHealthAsync()
    {
        if (_disposed || !_initialized) return;

        try
        {
            var healthTasks = _devices.Select(async device =>
            {
                try
                {
                    var healthInfo = await CollectDeviceHealthAsync(device);
                    _deviceHealth[device.Id] = healthInfo;

                    // Evaluate device health and take appropriate actions
                    await EvaluateDeviceHealth(device, healthInfo);
                    
                    _logger.LogDebug("Health info collected for device {DeviceId}. Health Score: {HealthScore:F2}, Status: {Status}", 
                        device.Id, healthInfo.HealthScore, healthInfo.IsHealthy ? "Healthy" : "Unhealthy");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to collect health info for device {DeviceId}", device.Id);
                }
            });

            await Task.WhenAll(healthTasks).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during device health monitoring");
        }
    }

    /// <summary>
    /// Updates load balancing weights and metrics
    /// </summary>
    private async Task UpdateLoadBalancingAsync()
    {
        if (_disposed || !_initialized) return;

        try
        {
            var loadTasks = _devices.Select(async device =>
            {
                try
                {
                    var loadInfo = await CollectDeviceLoadAsync(device);
                    _deviceLoad[device.Id] = loadInfo;

                    // Update selection weights based on current load
                    UpdateSelectionWeight(device, loadInfo);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to collect load info for device {DeviceId}", device.Id);
                }
            });

            await Task.WhenAll(loadTasks).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during load balancing update");
        }
    }

    /// <summary>
    /// Collects comprehensive health information for a device
    /// </summary>
    private Task<Orleans.GpuBridge.Abstractions.Models.DeviceHealthInfo> CollectDeviceHealthAsync(GpuDevice device)
    {
        var temperatureCelsius = 0;
        var maxTemperatureCelsius = 85;
        var isThermalThrottling = false;
        var powerUsageWatts = 0.0;
        var memoryUtilizationPercent = 0.0;
        var gpuUtilizationPercent = 0.0;
        var errorCount = 0;
        var status = DeviceStatus.Available;

        // Collect thermal information if available
        if (device.ThermalInfo != null)
        {
            temperatureCelsius = device.ThermalInfo.TemperatureCelsius;
            maxTemperatureCelsius = device.ThermalInfo.MaxTemperatureCelsius;
            isThermalThrottling = device.ThermalInfo.IsThrottling;
        }

        // Collect performance metrics if available
        if (device.PerformanceMetrics != null)
        {
            powerUsageWatts = device.PerformanceMetrics.PowerUsageWatts;
            memoryUtilizationPercent = device.PerformanceMetrics.MemoryUtilizationPercent;
            gpuUtilizationPercent = device.PerformanceMetrics.UtilizationPercent;
        }

        // Get work queue metrics
        if (_workQueues.TryGetValue(device.Index, out var queue))
        {
            var metrics = queue.GetMetrics();
            errorCount = (int)metrics.FailedItems;
            status = metrics.ErrorRate > 0.1 ? DeviceStatus.Error : DeviceStatus.Available;
        }

        var healthInfo = new Orleans.GpuBridge.Abstractions.Models.DeviceHealthInfo
        {
            DeviceId = device.Id,
            LastCheckTime = DateTime.UtcNow,
            TemperatureCelsius = temperatureCelsius,
            MaxTemperatureCelsius = maxTemperatureCelsius,
            IsThermalThrottling = isThermalThrottling,
            PowerUsageWatts = powerUsageWatts,
            MemoryUtilizationPercent = memoryUtilizationPercent,
            GpuUtilizationPercent = gpuUtilizationPercent,
            ErrorCount = errorCount,
            Status = status
        };

        return Task.FromResult(healthInfo);
    }

    /// <summary>
    /// Collects load balancing information for a device
    /// </summary>
    private Task<Orleans.GpuBridge.Abstractions.Models.DeviceLoadInfo> CollectDeviceLoadAsync(GpuDevice device)
    {
        var currentQueueDepth = 0;
        var successRatePercent = 100.0;

        // Get queue metrics
        if (_workQueues.TryGetValue(device.Index, out var queue))
        {
            var metrics = queue.GetMetrics();
            currentQueueDepth = metrics.QueuedItems;
            successRatePercent = metrics.ProcessedItems > 0 
                ? (1.0 - metrics.ErrorRate) * 100.0 
                : 100.0;
        }

        var loadInfo = new Orleans.GpuBridge.Abstractions.Models.DeviceLoadInfo
        {
            DeviceId = device.Id,
            LastUpdateTime = DateTime.UtcNow,
            CurrentQueueDepth = currentQueueDepth,
            SuccessRatePercent = successRatePercent,
            ProcessingRate = EstimateProcessingRate(device),
            AverageLatencyMs = EstimateAverageLatency(device)
        };

        return Task.FromResult(loadInfo);
    }

    /// <summary>
    /// Updates selection weight for load balancing
    /// </summary>
    private void UpdateSelectionWeight(GpuDevice device, DeviceLoadInfo loadInfo)
    {
        try
        {
            // Base weight starts at 1.0
            double weight = 1.0;

            // Reduce weight based on queue depth (higher queue = lower weight)
            weight *= Math.Max(0.1, 1.0 - (loadInfo.CurrentQueueDepth / 100.0));

            // Reduce weight based on error rate
            weight *= Math.Max(0.1, loadInfo.SuccessRatePercent / 100.0);

            // Reduce weight based on latency
            if (loadInfo.AverageLatencyMs > 100)
            {
                weight *= Math.Max(0.3, 1.0 - ((loadInfo.AverageLatencyMs - 100) / 1000.0));
            }

            // Apply health score if available (simplified for now)
            if (_deviceHealth.TryGetValue(device.Id, out var healthInfo))
            {
                // Apply health score to weight calculation
                var healthMultiplier = Math.Max(0.1, healthInfo.HealthScore);
                weight *= healthMultiplier;
                
                // Additional penalty for unhealthy devices
                if (!healthInfo.IsHealthy)
                {
                    weight *= 0.5; // Significant penalty for unhealthy devices
                }
                
                _logger.LogTrace("Applied health multiplier {HealthMultiplier:F2} to device {DeviceId}", 
                    healthMultiplier, device.Id);
            }

            // Update the load info with new weight
            var updatedLoadInfo = new Orleans.GpuBridge.Abstractions.Models.DeviceLoadInfo
            {
                DeviceId = loadInfo.DeviceId,
                LastUpdateTime = loadInfo.LastUpdateTime,
                CurrentQueueDepth = loadInfo.CurrentQueueDepth,
                SuccessRatePercent = loadInfo.SuccessRatePercent,
                ProcessingRate = loadInfo.ProcessingRate,
                AverageLatencyMs = loadInfo.AverageLatencyMs,
                SelectionWeight = Math.Max(0.05, weight)
            };
            _deviceLoad[device.Id] = updatedLoadInfo;

            _logger.LogTrace("Updated selection weight for device {DeviceId}: {Weight:F3}", 
                device.Id, weight);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to update selection weight for device {DeviceId}", device.Id);
        }
    }

    /// <summary>
    /// Estimates processing rate for a device
    /// </summary>
    private double EstimateProcessingRate(GpuDevice device)
    {
        // Base rate estimation based on device type and compute units
        return device.Type switch
        {
            DeviceType.CUDA => device.ComputeUnits * 0.1, // Conservative CUDA estimate
            DeviceType.OpenCL => device.ComputeUnits * 0.08, // OpenCL typically slower
            DeviceType.Metal => device.ComputeUnits * 0.09, // Metal performance estimate
            DeviceType.DirectCompute => device.ComputeUnits * 0.07,
            DeviceType.CPU => Environment.ProcessorCount * 0.5, // CPU parallel processing
            _ => 10.0 // Default fallback
        };
    }

    /// <summary>
    /// Estimates average latency for a device
    /// </summary>
    private double EstimateAverageLatency(GpuDevice device)
    {
        // Base latency estimation based on device type
        var baseLatency = device.Type switch
        {
            DeviceType.CUDA => 5.0, // CUDA typically low latency
            DeviceType.OpenCL => 10.0, // OpenCL higher overhead
            DeviceType.Metal => 8.0, // Metal moderate latency
            DeviceType.DirectCompute => 12.0, // DirectCompute higher overhead
            DeviceType.CPU => 2.0, // CPU lowest latency
            _ => 15.0 // Default conservative estimate
        };

        // Adjust based on current load
        if (_workQueues.TryGetValue(device.Index, out var queue))
        {
            var queueDepth = queue.QueuedItems;
            baseLatency += queueDepth * 2.0; // Linear latency increase with queue depth
        }

        return baseLatency;
    }

    /// <summary>
    /// Checks if device has required features for a workload
    /// </summary>
    private bool HasRequiredFeatures(GpuDevice device, IReadOnlyList<string> requiredFeatures)
    {
        if (requiredFeatures == null || !requiredFeatures.Any())
            return true;

        var deviceCapabilities = device.Capabilities ?? Array.Empty<string>();
        return requiredFeatures.All(required => 
            deviceCapabilities.Any(cap => cap.Contains(required, StringComparison.OrdinalIgnoreCase)));
    }

    /// <summary>
    /// Evaluates device health and takes appropriate actions
    /// </summary>
    private async Task EvaluateDeviceHealth(GpuDevice device, DeviceHealthInfo healthInfo)
    {
        try
        {
            // Log critical health issues
            if (!healthInfo.IsHealthy)
            {
                _logger.LogWarning("Device {DeviceId} is unhealthy. Health Score: {HealthScore:F2}, " +
                    "Temperature: {Temperature}°C, Thermal Throttling: {IsThermalThrottling}, " +
                    "Error Count: {ErrorCount}, Consecutive Failures: {ConsecutiveFailures}",
                    device.Id, healthInfo.HealthScore, healthInfo.TemperatureCelsius,
                    healthInfo.IsThermalThrottling, healthInfo.ErrorCount, healthInfo.ConsecutiveFailures);
            }

            // Handle thermal throttling
            if (healthInfo.IsThermalThrottling)
            {
                _logger.LogError("Device {DeviceId} is thermal throttling at {Temperature}°C. " +
                    "Max safe temperature: {MaxTemperature}°C",
                    device.Id, healthInfo.TemperatureCelsius, healthInfo.MaxTemperatureCelsius);
                
                // Could implement cooling strategies or reduce workload here
                await ImplementThermalThrottleResponse(device, healthInfo).ConfigureAwait(false);
            }

            // Handle excessive errors
            if (healthInfo.ErrorCount > 5)
            {
                _logger.LogError("Device {DeviceId} has {ErrorCount} errors. " +
                    "Consider device reset or removal from service",
                    device.Id, healthInfo.ErrorCount);
                    
                await HandleExcessiveErrors(device, healthInfo).ConfigureAwait(false);
            }

            // Handle consecutive failures
            if (healthInfo.ConsecutiveFailures >= 3)
            {
                _logger.LogError("Device {DeviceId} has {ConsecutiveFailures} consecutive failures. " +
                    "Temporarily removing from load balancing",
                    device.Id, healthInfo.ConsecutiveFailures);
                    
                await HandleConsecutiveFailures(device, healthInfo).ConfigureAwait(false);
            }

            // Handle high predicted failure probability
            if (healthInfo.PredictedFailureProbability > 0.8)
            {
                _logger.LogWarning("Device {DeviceId} has high failure probability: {FailureProbability:F2}. " +
                    "Consider proactive maintenance",
                    device.Id, healthInfo.PredictedFailureProbability);
            }

            // Update device status based on health
            await UpdateDeviceStatusBasedOnHealth(device, healthInfo).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to evaluate health for device {DeviceId}", device.Id);
        }
    }

    /// <summary>
    /// Implements response to thermal throttling
    /// </summary>
    private Task ImplementThermalThrottleResponse(GpuDevice device, DeviceHealthInfo healthInfo)
    {
        // Reduce device priority in load balancing
        if (_deviceLoad.TryGetValue(device.Id, out var loadInfo))
        {
            var throttledWeight = Math.Min(0.1, loadInfo.SelectionWeight * 0.5);
            var updatedLoadInfo = loadInfo with { SelectionWeight = throttledWeight };
            _deviceLoad[device.Id] = updatedLoadInfo;
            
            _logger.LogInformation("Reduced selection weight for thermally throttling device {DeviceId} to {Weight:F3}",
                device.Id, throttledWeight);
        }
        
        return Task.CompletedTask;
    }

    /// <summary>
    /// Handles devices with excessive error rates
    /// </summary>
    private Task HandleExcessiveErrors(GpuDevice device, DeviceHealthInfo healthInfo)
    {
        // Temporarily disable device if errors are too high
        if (healthInfo.ErrorCount > 10)
        {
            _logger.LogError("Disabling device {DeviceId} due to excessive errors: {ErrorCount}",
                device.Id, healthInfo.ErrorCount);
            
            // Set weight to minimum to avoid selection
            if (_deviceLoad.TryGetValue(device.Id, out var loadInfo))
            {
                var disabledLoadInfo = loadInfo with { SelectionWeight = 0.001 };
                _deviceLoad[device.Id] = disabledLoadInfo;
            }
        }
        
        return Task.CompletedTask;
    }

    /// <summary>
    /// Handles devices with consecutive failures
    /// </summary>
    private Task HandleConsecutiveFailures(GpuDevice device, DeviceHealthInfo healthInfo)
    {
        // Temporarily quarantine device
        if (_deviceLoad.TryGetValue(device.Id, out var loadInfo))
        {
            var quarantinedWeight = 0.01; // Very low weight
            var quarantineTime = DateTime.UtcNow.AddMinutes(5 * healthInfo.ConsecutiveFailures); // Escalating quarantine
            
            var quarantinedLoadInfo = loadInfo with 
            { 
                SelectionWeight = quarantinedWeight,
                ThrottleUntil = quarantineTime
            };
            _deviceLoad[device.Id] = quarantinedLoadInfo;
            
            _logger.LogWarning("Quarantined device {DeviceId} until {QuarantineTime} due to {ConsecutiveFailures} consecutive failures",
                device.Id, quarantineTime, healthInfo.ConsecutiveFailures);
        }
        
        return Task.CompletedTask;
    }

    /// <summary>
    /// Updates device status based on health evaluation
    /// </summary>
    private Task UpdateDeviceStatusBasedOnHealth(GpuDevice device, DeviceHealthInfo healthInfo)
    {
        // This would typically update device status in persistent storage
        // For now, we'll just log status changes
        
        if (healthInfo.HealthScore < 0.3)
        {
            _logger.LogError("Device {DeviceId} health score critically low: {HealthScore:F2}",
                device.Id, healthInfo.HealthScore);
        }
        else if (healthInfo.HealthScore < 0.7)
        {
            _logger.LogWarning("Device {DeviceId} health score degraded: {HealthScore:F2}",
                device.Id, healthInfo.HealthScore);
        }
        
        return Task.CompletedTask;
    }
}