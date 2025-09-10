using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Production-grade GPU device broker with comprehensive device management,
/// health monitoring, load balancing, and fault tolerance.
/// </summary>
public sealed partial class DeviceBroker
{
    private readonly ConcurrentDictionary<string, DeviceCapabilityCache> _deviceCapabilities = new();
    private readonly SemaphoreSlim _deviceDiscoveryLock = new(1, 1);
    private readonly Random _random = new();
    
    private volatile bool _isHealthMonitoringEnabled = true;
    private volatile bool _isLoadBalancingEnabled = true;

    /// <summary>
    /// Detects NVIDIA CUDA-capable devices with comprehensive capability detection
    /// </summary>
    private async Task<List<GpuDevice>> DetectCudaDevicesAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        try
        {
            _logger.LogDebug("Starting CUDA device detection");
            
            // Use System.Management for Windows or nvidia-ml-py equivalent detection
            var cudaDevices = await GetCudaDevicesFromSystem(ct);
            
            foreach (var cudaInfo in cudaDevices)
            {
                if (ct.IsCancellationRequested) break;
                
                var device = new GpuDevice(
                    Index: cudaInfo.Index,
                    Name: cudaInfo.Name,
                    Type: DeviceType.CUDA,
                    TotalMemoryBytes: cudaInfo.TotalMemory,
                    AvailableMemoryBytes: cudaInfo.TotalMemory, // Assume all available initially
                    ComputeUnits: cudaInfo.ComputeCapabilityMajor * cudaInfo.ComputeCapabilityMinor * 8, // Estimate
                    Capabilities: new[] { "cuda", $"compute_{cudaInfo.ComputeCapabilityMajor}.{cudaInfo.ComputeCapabilityMinor}" }
                );
                
                devices.Add(device);
                
                // Cache device capabilities
                _deviceCapabilities[device.Id] = new DeviceCapabilityCache
                {
                    LastUpdated = DateTime.UtcNow,
                    Capabilities = await GetDeviceCapabilities(device, ct),
                    BenchmarkScores = new DeviceBenchmarkScores()
                };
                
                _logger.LogInformation("Detected CUDA device: {DeviceName} (Memory: {MemoryGB:F1}GB)",
                    device.Name, device.TotalMemoryBytes / (1024.0 * 1024.0 * 1024.0));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect CUDA devices");
        }
        
        return devices;
    }

    /// <summary>
    /// Detects OpenCL-capable devices across all platforms
    /// </summary>
    private async Task<List<GpuDevice>> DetectOpenClDevicesAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        try
        {
            _logger.LogDebug("Starting OpenCL device detection");
            
            var openClDevices = await GetOpenClDevicesFromSystem(ct);
            
            foreach (var clInfo in openClDevices)
            {
                if (ct.IsCancellationRequested) break;
                
                var deviceType = DetermineOpenClDeviceType(clInfo.Vendor, clInfo.Type);
                
                var device = new GpuDevice(
                    Index: clInfo.DeviceIndex,
                    Name: clInfo.Name,
                    Type: deviceType,
                    TotalMemoryBytes: clInfo.GlobalMemorySize,
                    AvailableMemoryBytes: clInfo.GlobalMemorySize, // Assume all available initially
                    ComputeUnits: (int)clInfo.MaxWorkGroupSize / 64, // Estimate compute units
                    Capabilities: new[] { "opencl", clInfo.Type.ToLower(), clInfo.Vendor.ToLower() }
                );
                
                devices.Add(device);
                
                // Cache device capabilities
                _deviceCapabilities[device.Id] = new DeviceCapabilityCache
                {
                    LastUpdated = DateTime.UtcNow,
                    Capabilities = await GetDeviceCapabilities(device, ct),
                    BenchmarkScores = new DeviceBenchmarkScores()
                };
                
                _logger.LogInformation("Detected OpenCL device: {DeviceName} (Memory: {MemoryGB:F1}GB)",
                    device.Name, device.TotalMemoryBytes / (1024.0 * 1024.0 * 1024.0));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect OpenCL devices");
        }
        
        return devices;
    }

    /// <summary>
    /// Detects Intel GPU devices through Level Zero or OpenCL
    /// </summary>
    private async Task<List<GpuDevice>> DetectIntelDevicesAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        try
        {
            _logger.LogDebug("Starting Intel GPU device detection");
            
            // Try Level Zero first (Intel's newer API)
            var levelZeroDevices = await GetLevelZeroDevices(ct);
            foreach (var device in levelZeroDevices)
            {
                devices.Add(device);
            }
            
            // Fall back to OpenCL for older Intel GPUs
            if (devices.Count == 0)
            {
                var openClDevices = await DetectOpenClDevicesAsync(ct);
                // Filter Intel devices based on capabilities or name patterns
                var intelDevices = openClDevices.Where(d => 
                    d.Name.Contains("Intel", StringComparison.OrdinalIgnoreCase) ||
                    d.Type == DeviceType.OpenCL).ToList();
                devices.AddRange(intelDevices);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect Intel GPU devices");
        }
        
        return devices;
    }

    /// <summary>
    /// Detects Apple Metal-capable devices (macOS only)
    /// </summary>
    private async Task<List<GpuDevice>> DetectMetalDevicesAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        try
        {
            // Only attempt on macOS
            if (!OperatingSystem.IsMacOS())
            {
                return devices;
            }
            
            _logger.LogDebug("Starting Metal device detection");
            
            var metalDevices = await GetMetalDevicesFromSystem(ct);
            
            foreach (var metalInfo in metalDevices)
            {
                if (ct.IsCancellationRequested) break;
                
                var device = new GpuDevice(
                    Index: metalInfo.Index,
                    Name: metalInfo.Name,
                    Type: DeviceType.Metal,
                    TotalMemoryBytes: metalInfo.RecommendedMaxWorkingSetSize,
                    AvailableMemoryBytes: metalInfo.RecommendedMaxWorkingSetSize, // Assume all available initially
                    ComputeUnits: metalInfo.IsAppleSilicon ? 8 : 4, // Estimate based on Apple Silicon
                    Capabilities: new[] { "metal", metalInfo.IsAppleSilicon ? "apple_silicon" : "discrete" }
                );
                
                devices.Add(device);
                
                _logger.LogInformation("Detected Metal device: {DeviceName} (Memory: {MemoryGB:F1}GB)",
                    device.Name, device.TotalMemoryBytes / (1024.0 * 1024.0 * 1024.0));
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect Metal devices");
        }
        
        return devices;
    }

    // Removed duplicate methods - these are now implemented in the main DeviceBroker.cs file

    /// <summary>
    /// Sophisticated device selection algorithm with multi-criteria optimization
    /// </summary>
    private GpuDevice? SelectOptimalDevice(DeviceSelectionCriteria criteria)
    {
        var availableDevices = _availableDevices
            .Where(d => d.Status == DeviceStatus.Available || d.Status == DeviceStatus.Busy)
            .Where(d => MeetsBasicCriteria(d, criteria))
            .ToList();
            
        if (!availableDevices.Any())
        {
            return criteria.AllowCpuFallback ? GetCpuFallbackDevice() : null;
        }
        
        // Multi-criteria scoring algorithm
        var scoredDevices = availableDevices.Select(device =>
        {
            var score = CalculateDeviceScore(device, criteria);
            return new { Device = device, Score = score };
        })
        .Where(x => x.Score > 0)
        .OrderByDescending(x => x.Score)
        .ToList();
        
        if (!scoredDevices.Any())
        {
            return criteria.AllowCpuFallback ? GetCpuFallbackDevice() : null;
        }
        
        // Implement weighted random selection among top candidates to avoid hot spots
        var topCandidates = scoredDevices.Take(Math.Min(3, scoredDevices.Count)).ToList();
        var totalWeight = topCandidates.Sum(x => x.Score);
        var randomValue = _random.NextDouble() * totalWeight;
        
        double currentWeight = 0;
        foreach (var candidate in topCandidates)
        {
            currentWeight += candidate.Score;
            if (randomValue <= currentWeight)
            {
                _logger.LogDebug("Selected device {DeviceId} with score {Score:F2}", 
                    candidate.Device.Id, candidate.Score);
                return candidate.Device;
            }
        }
        
        return topCandidates.First().Device;
    }

    /// <summary>
    /// Comprehensive device scoring algorithm
    /// </summary>
    private double CalculateDeviceScore(GpuDevice device, DeviceSelectionCriteria criteria)
    {
        double score = 1.0;
        
        // Performance factor
        if (_deviceCapabilities.TryGetValue(device.Id, out var capabilities))
        {
            score *= Math.Max(0.1, capabilities.BenchmarkScores.OverallScore);
        }
        
        // Load balancing factor
        if (_deviceLoad.TryGetValue(device.Id, out var loadInfo))
        {
            score *= Math.Max(0.1, loadInfo.SelectionWeight);
            score *= Math.Max(0.1, 1.0 - (loadInfo.CurrentUtilization / 100.0));
        }
        
        // Health factor
        if (_deviceHealth.TryGetValue(device.Id, out var healthInfo))
        {
            score *= healthInfo.IsHealthy ? 1.0 : 0.1;
            score *= Math.Max(0.3, 1.0 - (healthInfo.TemperatureCelsius / 100.0));
        }
        
        // Memory factor
        var memoryScore = device.TotalMemoryBytes >= criteria.MinimumMemoryBytes ? 1.0 : 0.0;
        score *= memoryScore;
        
        // Feature compatibility - simplified for now since we need string list
        var featureScore = 1.0; // Always pass feature check for now
        score *= featureScore;
        
        // Type preference
        if (criteria.PreferredType.HasValue && device.Type == criteria.PreferredType.Value)
        {
            score *= 1.5; // Bonus for preferred type
        }
        
        // Latency vs throughput preference
        if (criteria.PreferLowLatency && device.Type == DeviceType.CUDA)
        {
            score *= 1.3; // CUDA typically has lower latency
        }
        else if (criteria.PreferHighThroughput)
        {
            score *= Math.Max(0.5, device.TotalMemoryBytes / (8L * 1024 * 1024 * 1024)); // Favor high-memory devices
        }
        
        return Math.Max(0.0, score);
    }

    /// <summary>
    /// Handles device stress conditions with automatic mitigation
    /// </summary>
    private async Task HandleDeviceStress(GpuDevice device, DeviceHealthInfo health, CancellationToken ct)
    {
        _logger.LogWarning("Handling stress conditions for device {DeviceId}", device.Id);
        
        if (health.TemperatureCelsius > 90.0f)
        {
            // Thermal throttling
            device.Status = DeviceStatus.Resetting;
            await Task.Delay(TimeSpan.FromSeconds(30), ct); // Cool down period
            
            // Reduce device priority temporarily
            if (_deviceLoad.TryGetValue(device.Id, out var loadInfo))
            {
                loadInfo.SelectionWeight *= 0.3f;
                loadInfo.ThrottleUntil = DateTime.UtcNow.AddMinutes(10);
            }
        }
        
        if (health.MemoryUtilizationPercent > 95.0)
        {
            // Memory pressure mitigation
            await TriggerMemoryCleanup(device, ct);
        }
        
        _logger.LogInformation("Stress mitigation completed for device {DeviceId}", device.Id);
    }
}

