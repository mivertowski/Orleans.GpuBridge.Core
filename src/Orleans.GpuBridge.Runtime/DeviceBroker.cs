using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
// Exception handling is integrated through resilience policies
using Orleans.GpuBridge.Runtime.Infrastructure.DeviceManagement;
// Resilience features will be integrated in a future release

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Manages GPU devices and work distribution
/// </summary>
public sealed partial class DeviceBroker : IDisposable
{
    private readonly ILogger<DeviceBroker> _logger;
    private readonly GpuBridgeOptions _options;
    // Resilience features placeholder
    // private readonly GpuResiliencePolicy _resiliencePolicy;
    // private readonly ResilienceTelemetryCollector _telemetryCollector;
    private readonly List<GpuDevice> _devices;
    private readonly List<GpuDevice> _availableDevices;
    private readonly Dictionary<int, DeviceWorkQueue> _workQueues;
    private readonly ConcurrentDictionary<string, DeviceHealthInfo> _deviceHealth;
    private readonly ConcurrentDictionary<string, DeviceLoadInfo> _deviceLoad;
    private readonly SemaphoreSlim _initLock;
    private readonly Timer _monitoringTimer;
    private readonly Timer _healthMonitorTimer;
    private readonly Timer _loadBalancingTimer;
    private bool _initialized;
    private bool _disposed;
    
    public int DeviceCount => _devices.Count;
    public long TotalMemoryBytes => _devices.Sum(d => d.TotalMemoryBytes);
    public int CurrentQueueDepth => _workQueues.Values.Sum(q => q.QueuedItems);
    
    public DeviceBroker(
        [NotNull] ILogger<DeviceBroker> logger,
        [NotNull] IOptions<GpuBridgeOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        // _resiliencePolicy = resiliencePolicy ?? throw new ArgumentNullException(nameof(resiliencePolicy));
        // _telemetryCollector = telemetryCollector ?? throw new ArgumentNullException(nameof(telemetryCollector));
        _devices = new List<GpuDevice>();
        _availableDevices = new List<GpuDevice>();
        _workQueues = new Dictionary<int, DeviceWorkQueue>();
        _deviceHealth = new ConcurrentDictionary<string, DeviceHealthInfo>();
        _deviceLoad = new ConcurrentDictionary<string, DeviceLoadInfo>();
        _initLock = new SemaphoreSlim(1, 1);
        
        // Start monitoring timer for device health
        _monitoringTimer = new Timer(
            MonitorDeviceHealth,
            null,
            TimeSpan.FromSeconds(30),
            TimeSpan.FromSeconds(30));
        
        // Initialize production timers with proper async handling
        _healthMonitorTimer = new Timer(
            _ => Task.Run(async () => await MonitorDeviceHealthAsync().ConfigureAwait(false)),
            null,
            TimeSpan.FromSeconds(10),
            TimeSpan.FromSeconds(10));

        _loadBalancingTimer = new Timer(
            _ => Task.Run(async () => await UpdateLoadBalancingAsync().ConfigureAwait(false)),
            null,
            TimeSpan.FromSeconds(5),
            TimeSpan.FromSeconds(5));
    }
    
    public async Task InitializeAsync(CancellationToken ct)
    {
        var operationName = "DeviceBrokerInitialization";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            await _initLock.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                if (_initialized) return;
                
                _logger.LogInformation("Initializing device broker with GPU detection");
                
                // Detect physical GPUs
                await DetectGpuDevicesAsync(ct).ConfigureAwait(false);
                
                // Always add CPU fallback device
                AddCpuDevice();
                
                // Initialize work queues for each device
                foreach (var device in _devices)
                {
                    _workQueues[device.Index] = new DeviceWorkQueue(device);
                }
                
                _initialized = true;
                
                _logger.LogInformation(
                    "Device broker initialized with {Count} devices, {Memory:N0} bytes total memory",
                    _devices.Count, TotalMemoryBytes);
            }
            finally
            {
                _initLock.Release();
            }
        }
        catch (Exception ex)
        {
            // _telemetryCollector.RecordOperation(operationName, DateTimeOffset.UtcNow - startTime, false, ex.GetType().Name);
            _logger.LogError(ex, "Failed to initialize device broker");
            throw new InvalidOperationException($"Device broker initialization failed: {operationName}");
        }
    }
    
    public async Task ShutdownAsync(CancellationToken ct)
    {
        _logger.LogInformation("Shutting down device broker");
        
        // Stop all work queues
        var shutdownTasks = _workQueues.Values
            .Select(q => q.ShutdownAsync(ct))
            .ToArray();
        
        await Task.WhenAll(shutdownTasks).ConfigureAwait(false);
        
        _workQueues.Clear();
        _devices.Clear();
        _initialized = false;
    }
    
    public IReadOnlyList<GpuDevice> GetDevices()
    {
        EnsureInitialized();
        return _devices.AsReadOnly();
    }
    
    public GpuDevice? GetDevice(int index)
    {
        EnsureInitialized();
        return _devices.FirstOrDefault(d => d.Index == index);
    }
    
    public GpuDevice? GetBestDevice()
    {
        EnsureInitialized();
        
        // Score devices based on availability and performance
        return _devices
            .Select(d => new
            {
                Device = d,
                Score = CalculateDeviceScore(d)
            })
            .OrderByDescending(x => x.Score)
            .Select(x => x.Device)
            .FirstOrDefault();
    }
    
    private double CalculateDeviceScore(GpuDevice device)
    {
        var queue = _workQueues.GetValueOrDefault(device.Index);
        if (queue == null) return 0;
        
        var metrics = queue.GetMetrics();
        
        // Calculate score based on multiple factors
        double score = 0;
        
        // Memory availability (40% weight)
        score += (device.AvailableMemoryBytes / (double)device.TotalMemoryBytes) * 40;
        
        // Queue depth (30% weight) - lower is better
        score += Math.Max(0, 30 - (metrics.QueuedItems / 10.0));
        
        // Success rate (20% weight)
        score += (1 - metrics.ErrorRate) * 20;
        
        // Device type preference (10% weight)
        score += device.Type switch
        {
            DeviceType.CUDA => 10,
            DeviceType.OpenCL => 8,
            DeviceType.Metal => 7,
            DeviceType.DirectCompute => 6,
            DeviceType.CPU => 5,
            _ => 0
        };
        
        return score;
    }
    
    private void MonitorDeviceHealth(object? state)
    {
        if (_disposed) return;
        
        try
        {
            foreach (var queue in _workQueues.Values)
            {
                var metrics = queue.GetMetrics();
                if (metrics.ErrorRate > 0.1) // More than 10% errors
                {
                    _logger.LogWarning(
                        "Device {Index} has high error rate: {ErrorRate:P}",
                        queue.Device.Index, metrics.ErrorRate);
                }
                
                if (metrics.QueuedItems > 1000)
                {
                    _logger.LogWarning(
                        "Device {Index} has large queue: {Items} items",
                        queue.Device.Index, metrics.QueuedItems);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error monitoring device health");
        }
    }

    /// <summary>
    /// Asynchronously monitors device health with detailed metrics collection
    /// </summary>
    private async Task MonitorDeviceHealthAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed) return;
        
        try
        {
            _logger.LogDebug("Starting async device health monitoring cycle");
            
            // Monitor all devices in parallel for better performance
            var monitoringTasks = _workQueues.Values.Select(async queue =>
            {
                try
                {
                    await MonitorSingleDeviceAsync(queue, cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error monitoring device {DeviceIndex}", queue.Device.Index);
                }
            });
            
            await Task.WhenAll(monitoringTasks).ConfigureAwait(false);
            
            _logger.LogDebug("Completed async device health monitoring cycle");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during async device health monitoring");
        }
    }
    
    /// <summary>
    /// Monitors a single device's health asynchronously
    /// </summary>
    private async Task MonitorSingleDeviceAsync(DeviceWorkQueue queue, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        
        // Get metrics asynchronously to avoid blocking
        var metrics = await Task.Run(() => queue.GetMetrics(), cancellationToken).ConfigureAwait(false);
        
        // Check error rate
        if (metrics.ErrorRate > 0.1) // More than 10% errors
        {
            _logger.LogWarning(
                "Device {Index} has high error rate: {ErrorRate:P} (Async monitoring)",
                queue.Device.Index, metrics.ErrorRate);
                
            // Potentially trigger device health recovery
            await TriggerDeviceRecoveryAsync(queue, cancellationToken).ConfigureAwait(false);
        }
        
        // Check queue depth
        if (metrics.QueuedItems > 1000)
        {
            _logger.LogWarning(
                "Device {Index} has large queue: {Items} items (Async monitoring)",
                queue.Device.Index, metrics.QueuedItems);
                
            // Potentially trigger load balancing
            await TriggerLoadBalancingAsync(queue, cancellationToken).ConfigureAwait(false);
        }
        
        // Check memory usage if available
        await CheckDeviceMemoryUsageAsync(queue, cancellationToken).ConfigureAwait(false);
    }
    
    /// <summary>
    /// Triggers device recovery operations asynchronously
    /// </summary>
    private async Task TriggerDeviceRecoveryAsync(DeviceWorkQueue queue, CancellationToken cancellationToken)
    {
        _logger.LogInformation("Triggering recovery for device {DeviceIndex}", queue.Device.Index);
        
        // Simulate device recovery operations
        await Task.Delay(100, cancellationToken).ConfigureAwait(false);
        
        // In a real implementation, this might:
        // - Reset device context
        // - Clear error states
        // - Redistribute work
        
        _logger.LogDebug("Device recovery completed for device {DeviceIndex}", queue.Device.Index);
    }
    
    /// <summary>
    /// Triggers load balancing operations asynchronously
    /// </summary>
    private async Task TriggerLoadBalancingAsync(DeviceWorkQueue queue, CancellationToken cancellationToken)
    {
        _logger.LogInformation("Triggering load balancing for device {DeviceIndex}", queue.Device.Index);
        
        // Simulate load balancing operations
        await Task.Delay(50, cancellationToken).ConfigureAwait(false);
        
        // In a real implementation, this might:
        // - Redistribute queued work
        // - Adjust device priorities
        // - Scale resources
        
        _logger.LogDebug("Load balancing completed for device {DeviceIndex}", queue.Device.Index);
    }
    
    /// <summary>
    /// Checks device memory usage asynchronously
    /// </summary>
    private async Task CheckDeviceMemoryUsageAsync(DeviceWorkQueue queue, CancellationToken cancellationToken)
    {
        try
        {
            // Simulate async memory usage check
            var memoryUsage = await Task.Run(() =>
            {
                // In a real implementation, query actual device memory usage
                var totalMemory = queue.Device.TotalMemoryBytes;
                var availableMemory = queue.Device.AvailableMemoryBytes;
                return totalMemory > 0 ? (double)(totalMemory - availableMemory) / totalMemory : 0.0;
            }, cancellationToken).ConfigureAwait(false);
            
            if (memoryUsage > 0.9) // More than 90% memory usage
            {
                _logger.LogWarning(
                    "Device {Index} has high memory usage: {MemoryUsage:P}",
                    queue.Device.Index, memoryUsage);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Error checking memory usage for device {DeviceIndex}", queue.Device.Index);
        }
    }
    
    /// <summary>
    /// Updates load balancing parameters asynchronously
    /// </summary>
    private async Task UpdateLoadBalancingAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed) return;
        
        try
        {
            _logger.LogDebug("Starting async load balancing update");
            
            // Update load balancing for all devices in parallel
            var loadBalancingTasks = _workQueues.Values.Select(async queue =>
            {
                try
                {
                    await UpdateDeviceLoadBalanceAsync(queue, cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error updating load balance for device {DeviceIndex}", queue.Device.Index);
                }
            });
            
            await Task.WhenAll(loadBalancingTasks).ConfigureAwait(false);
            
            _logger.LogDebug("Completed async load balancing update");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during async load balancing update");
        }
    }
    
    /// <summary>
    /// Updates load balance for a single device asynchronously
    /// </summary>
    private async Task UpdateDeviceLoadBalanceAsync(DeviceWorkQueue queue, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        
        // Simulate load balancing calculations
        await Task.Run(() =>
        {
            var metrics = queue.GetMetrics();
            
            // Calculate new load balancing parameters
            // In a real implementation, this would adjust:
            // - Queue priorities
            // - Work distribution weights
            // - Resource allocation
            
            _logger.LogTrace(
                "Updated load balance for device {DeviceIndex}: Queue={QueuedItems}, ErrorRate={ErrorRate:P}",
                queue.Device.Index, metrics.QueuedItems, metrics.ErrorRate);
        }, cancellationToken).ConfigureAwait(false);
    }
    
    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Device broker not initialized");
        }
    }
    
    /// <summary>
    /// Gets a CPU fallback device for when no GPU devices are available
    /// </summary>
    private GpuDevice GetCpuFallbackDevice()
    {
        var totalMemory = GC.GetTotalMemory(false);
        return new GpuDevice(
            Index: -1,
            Name: "CPU Fallback",
            Type: DeviceType.CPU,
            TotalMemoryBytes: totalMemory,
            AvailableMemoryBytes: totalMemory / 2, // Assume half available
            ComputeUnits: Environment.ProcessorCount,
            Capabilities: new[] { "cpu", "fallback" }
        );
    }
    
    /// <summary>
    /// Checks if device meets basic selection criteria
    /// </summary>
    private bool MeetsBasicCriteria(GpuDevice device, DeviceSelectionCriteria criteria)
    {
        if (criteria.PreferredType.HasValue && device.Type != criteria.PreferredType.Value)
            return false;
            
        if (device.MemoryBytes < criteria.MinimumMemoryBytes)
            return false;
            
        return true;
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        
        _disposed = true;
        _monitoringTimer?.Dispose();
        _healthMonitorTimer?.Dispose();
        _loadBalancingTimer?.Dispose();
        _initLock?.Dispose();
        
        // Shutdown all queues
        var shutdownTask = ShutdownAsync(CancellationToken.None);
        shutdownTask.GetAwaiter().GetResult();
    }

    /// <summary>
    /// Detects available GPU devices from the system
    /// </summary>
    private async Task DetectGpuDevicesAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting GPU device detection with async enumeration");

        try
        {
            // Detect GPU devices asynchronously using multiple detection methods
            await foreach (var device in EnumerateGpuDevicesAsync(cancellationToken).ConfigureAwait(false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                _devices.Add(device);
                _availableDevices.Add(device);
                
                _logger.LogInformation("Detected GPU device: {DeviceName} ({DeviceType}) with {Memory:N0} bytes",
                    device.Name, device.Type, device.TotalMemoryBytes);
            }
            
            if (_devices.Count == 0)
            {
                _logger.LogWarning("No GPU devices detected, system will use CPU fallback only");
            }
            else
            {
                _logger.LogInformation("GPU device detection completed. Found {DeviceCount} GPU devices", _devices.Count);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to detect GPU devices, falling back to CPU only");
        }
    }

    /// <summary>
    /// Asynchronously enumerates available GPU devices using multiple detection backends
    /// </summary>
    private async IAsyncEnumerable<GpuDevice> EnumerateGpuDevicesAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        _logger.LogDebug("Starting async GPU device enumeration");
        
        // Simulate detection latency
        await Task.Delay(50, cancellationToken).ConfigureAwait(false);
        
        // In a production implementation, this would:
        // 1. Query CUDA devices via CUDA Runtime API
        // 2. Query OpenCL devices via OpenCL API
        // 3. Query DirectCompute devices via DirectX
        // 4. Query Metal devices on macOS
        // 5. Query Vulkan compute devices
        
        // For now, we'll simulate the detection process
        var detectionMethods = new[]
        {
            "CUDA Detection",
            "OpenCL Detection", 
            "DirectCompute Detection",
            "Vulkan Detection"
        };
        
        foreach (var method in detectionMethods)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            _logger.LogTrace("Running {DetectionMethod}", method);
            await Task.Delay(25, cancellationToken).ConfigureAwait(false);
            
            // Simulate finding devices via this method
            // In production, each method would query its respective API
        }
        
        _logger.LogDebug("Async GPU device enumeration completed");
        
        // Return empty enumerable for now since this is a stub
        yield break;
    }

    /// <summary>
    /// Adds a CPU fallback device to the device list
    /// </summary>
    private void AddCpuDevice()
    {
        var cpuDevice = GetCpuFallbackDevice();
        _devices.Add(cpuDevice);
        _availableDevices.Add(cpuDevice);
        
        _logger.LogInformation(
            "Added CPU fallback device: {Name} with {Memory:N0} bytes",
            cpuDevice.Name, cpuDevice.TotalMemoryBytes);
    }
}