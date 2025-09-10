using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

/// <summary>
/// ILGPU device manager implementation
/// </summary>
internal sealed class ILGPUDeviceManager : IDeviceManager
{
    private readonly ILogger<ILGPUDeviceManager> _logger;
    private readonly Context _context;
    private readonly List<ILGPUComputeDevice> _devices;
    private readonly Dictionary<IComputeDevice, ILGPUComputeContext> _contexts;
    private readonly Dictionary<string, long> _kernelsExecuted;
    private readonly Dictionary<string, long> _bytesTransferred;
    private readonly object _metricsLock = new object();
    private bool _initialized;
    private bool _disposed;

    public ILGPUDeviceManager(ILogger<ILGPUDeviceManager> logger, Context context)
    {
        _logger = logger;
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _devices = new List<ILGPUComputeDevice>();
        _contexts = new Dictionary<IComputeDevice, ILGPUComputeContext>();
        _kernelsExecuted = new Dictionary<string, long>();
        _bytesTransferred = new Dictionary<string, long>();
    }

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_initialized)
        {
            _logger.LogWarning("ILGPU device manager already initialized");
            return;
        }

        _logger.LogInformation("Discovering ILGPU compute devices");

        try
        {
            // Discover all available accelerators asynchronously
            var accelerators = await DiscoverAcceleratorsAsync(cancellationToken).ConfigureAwait(false);

            // Create device wrappers
            for (int i = 0; i < accelerators.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var accelerator = accelerators[i];
                var device = new ILGPUComputeDevice(i, accelerator, _logger);
                _devices.Add(device);
            }

            _initialized = true;

            _logger.LogInformation(
                "ILGPU device manager initialized with {DeviceCount} devices",
                _devices.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize ILGPU device manager");
            throw;
        }
    }

    public IReadOnlyList<IComputeDevice> GetDevices()
    {
        EnsureInitialized();
        return _devices.Cast<IComputeDevice>().ToList();
    }

    public IComputeDevice? GetDevice(int deviceIndex)
    {
        EnsureInitialized();
        return deviceIndex >= 0 && deviceIndex < _devices.Count ? _devices[deviceIndex] : null;
    }

    public IComputeDevice GetDefaultDevice()
    {
        EnsureInitialized();
        
        if (_devices.Count == 0)
        {
            throw new InvalidOperationException("No compute devices available");
        }

        // Prefer GPU devices over CPU
        var gpuDevice = _devices.FirstOrDefault(d => d.Type != DeviceType.CPU);
        return gpuDevice ?? _devices[0];
    }

    public IComputeDevice SelectDevice(DeviceSelectionCriteria criteria)
    {
        EnsureInitialized();

        if (_devices.Count == 0)
        {
            throw new InvalidOperationException("No compute devices available");
        }

        // Score devices based on criteria
        var scoredDevices = new List<(ILGPUComputeDevice Device, int Score)>();

        foreach (var device in _devices)
        {
            var score = CalculateDeviceScore(device, criteria);
            if (score >= 0) // -1 means device doesn't meet requirements
            {
                scoredDevices.Add((device, score));
            }
        }

        if (scoredDevices.Count == 0)
        {
            _logger.LogWarning("No devices meet the selection criteria, using default device");
            return GetDefaultDevice();
        }

        // Return device with highest score
        var selectedDevice = scoredDevices.OrderByDescending(d => d.Score).First().Device;

        _logger.LogDebug(
            "Selected device: {DeviceName} ({DeviceType}) with score {Score}",
            selectedDevice.Name,
            selectedDevice.Type,
            scoredDevices.First(d => d.Device == selectedDevice).Score);

        return selectedDevice;
    }

    public async Task<IComputeContext> CreateContextAsync(
        IComputeDevice device,
        ContextOptions options,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (device is not ILGPUComputeDevice ilgpuDevice)
        {
            throw new ArgumentException("Device must be an ILGPU compute device", nameof(device));
        }

        // Check if context already exists
        if (_contexts.TryGetValue(device, out var existingContext))
        {
            _logger.LogDebug("Returning existing context for device: {DeviceName}", device.Name);
            return existingContext;
        }

        // Create new context
        var context = new ILGPUComputeContext(ilgpuDevice, options, _logger);
        await context.InitializeAsync(cancellationToken);

        _contexts[device] = context;

        _logger.LogDebug("Created new context for device: {DeviceName}", device.Name);
        return context;
    }

    public async Task<DeviceMetrics> GetDeviceMetricsAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (device is not ILGPUComputeDevice ilgpuDevice)
        {
            throw new ArgumentException("Device must be an ILGPU compute device", nameof(device));
        }

        try
        {
            var accelerator = ilgpuDevice.Accelerator;
            
            // Basic metrics that ILGPU can provide
            // Note: ILGPU doesn't expose detailed memory info directly
            var totalMemory = ilgpuDevice.TotalMemoryBytes;
            var availableMemory = ilgpuDevice.AvailableMemoryBytes;
            var usedMemory = totalMemory - availableMemory;

            var metrics = new DeviceMetrics
            {
                GpuUtilizationPercent = 0, // ILGPU doesn't provide utilization directly
                MemoryUtilizationPercent = totalMemory > 0 ? (float)(usedMemory * 100.0 / totalMemory) : 0,
                UsedMemoryBytes = usedMemory,
                TemperatureCelsius = 0, // Not available via ILGPU
                PowerWatts = 0, // Not available via ILGPU
                FanSpeedPercent = 0, // Not available via ILGPU
                KernelsExecuted = GetKernelsExecutedForDevice(device.DeviceId),
                BytesTransferred = GetBytesTransferredForDevice(device.DeviceId),
                Uptime = TimeSpan.FromMilliseconds(Environment.TickCount64)
            };

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get metrics for device: {DeviceName}", device.Name);
            
            // Return default metrics on error
            return new DeviceMetrics
            {
                GpuUtilizationPercent = 0,
                MemoryUtilizationPercent = 0,
                UsedMemoryBytes = 0,
                TemperatureCelsius = 0,
                PowerWatts = 0,
                FanSpeedPercent = 0,
                KernelsExecuted = 0,
                BytesTransferred = 0,
                Uptime = TimeSpan.Zero
            };
        }
    }

    public async Task ResetDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (device is not ILGPUComputeDevice ilgpuDevice)
        {
            throw new ArgumentException("Device must be an ILGPU compute device", nameof(device));
        }

        try
        {
            _logger.LogInformation("Resetting device: {DeviceName}", device.Name);

            // Dispose existing context if any
            if (_contexts.TryGetValue(device, out var context))
            {
                context.Dispose();
                _contexts.Remove(device);
            }

            // Synchronize the accelerator asynchronously
            await Task.Run(() => ilgpuDevice.Accelerator.Synchronize(), cancellationToken).ConfigureAwait(false);

            // Clear memory with async GC operations
            await Task.Run(() =>
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("Device reset completed: {DeviceName}", device.Name);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reset device: {DeviceName}", device.Name);
            throw;
        }
    }

    private int CalculateDeviceScore(ILGPUComputeDevice device, DeviceSelectionCriteria criteria)
    {
        var score = 0;

        // Check minimum requirements first
        if (device.TotalMemoryBytes < criteria.MinMemoryBytes)
        {
            return -1; // Doesn't meet requirements
        }

        if (device.ComputeUnits < criteria.MinComputeUnits)
        {
            return -1; // Doesn't meet requirements
        }

        // Check excluded devices
        if (criteria.ExcludeDevices?.Contains(device.Index) == true)
        {
            return -1; // Explicitly excluded
        }

        // Score based on device type preference
        if (criteria.PreferredType.HasValue && device.Type == criteria.PreferredType.Value)
        {
            score += 1000;
        }

        // Prefer high-performance devices
        if (criteria.PreferHighestPerformance)
        {
            // Score based on compute units and memory
            score += device.ComputeUnits * 10;
            score += (int)(device.TotalMemoryBytes / (1024 * 1024)); // MB
            
            // Prefer GPU over CPU
            if (device.Type != DeviceType.CPU)
            {
                score += 500;
            }
        }

        // Check for unified memory requirement
        if (criteria.RequireUnifiedMemory)
        {
            // ILGPU supports unified memory on supported devices
            var supportsUnifiedMemory = device.Accelerator.AcceleratorType != AcceleratorType.CPU;
            if (!supportsUnifiedMemory)
            {
                score -= 100; // Penalty for not supporting unified memory
            }
        }

        // Check required feature
        if (criteria.RequiredFeature != DeviceFeatures.None)
        {
            if (!device.SupportsFeature(criteria.RequiredFeature.ToString()))
            {
                return -1; // Required feature not supported
            }
            score += 50;
        }

        return score;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Device manager not initialized");
        }
    }

    /// <summary>
    /// Gets the number of kernels executed for a specific device
    /// </summary>
    private long GetKernelsExecutedForDevice(string deviceId)
    {
        lock (_metricsLock)
        {
            return _kernelsExecuted.TryGetValue(deviceId, out var count) ? count : 0;
        }
    }

    /// <summary>
    /// Gets the number of bytes transferred for a specific device
    /// </summary>
    private long GetBytesTransferredForDevice(string deviceId)
    {
        lock (_metricsLock)
        {
            return _bytesTransferred.TryGetValue(deviceId, out var bytes) ? bytes : 0;
        }
    }

    /// <summary>
    /// Increments the kernel execution count for a device
    /// </summary>
    internal void IncrementKernelCount(string deviceId)
    {
        lock (_metricsLock)
        {
            _kernelsExecuted[deviceId] = _kernelsExecuted.TryGetValue(deviceId, out var count) ? count + 1 : 1;
        }
    }

    /// <summary>
    /// Adds to the bytes transferred count for a device
    /// </summary>
    internal void AddBytesTransferred(string deviceId, long bytes)
    {
        lock (_metricsLock)
        {
            _bytesTransferred[deviceId] = _bytesTransferred.TryGetValue(deviceId, out var existing) ? existing + bytes : bytes;
        }
    }

    /// <summary>
    /// Asynchronously discovers ILGPU accelerators with proper device enumeration
    /// </summary>
    private async Task<List<Accelerator>> DiscoverAcceleratorsAsync(CancellationToken cancellationToken)
    {
        var accelerators = new List<Accelerator>();

        await Task.Run(async () =>
        {
            // Enumerate all available devices and create accelerators
            foreach (var device in _context)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    // Create accelerator with device-specific initialization
                    var accelerator = await CreateAcceleratorAsync(device, cancellationToken).ConfigureAwait(false);
                    if (accelerator != null)
                    {
                        accelerators.Add(accelerator);
                        
                        var deviceType = accelerator.AcceleratorType switch
                        {
                            AcceleratorType.Cuda => "CUDA",
                            AcceleratorType.OpenCL => "OpenCL", 
                            AcceleratorType.CPU => "CPU",
                            _ => "Unknown"
                        };
                        
                        _logger.LogDebug("Created {DeviceType} accelerator: {Name}", deviceType, accelerator.Name);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Failed to create accelerator for device: {DeviceName}", 
                        device.Name);
                }
            }

            // Ensure we have at least one accelerator (CPU fallback)
            if (accelerators.Count == 0)
            {
                var cpuAccelerator = await CreateCpuFallbackAcceleratorAsync(cancellationToken).ConfigureAwait(false);
                if (cpuAccelerator != null)
                {
                    accelerators.Add(cpuAccelerator);
                }
            }
        }, cancellationToken).ConfigureAwait(false);

        return accelerators;
    }

    /// <summary>
    /// Creates an accelerator for a specific device with async initialization
    /// </summary>
    private async Task<Accelerator?> CreateAcceleratorAsync(Device device, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                return device.CreateAccelerator(_context);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Failed to create accelerator for device {DeviceName}", device.Name);
                return null;
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Creates a CPU fallback accelerator asynchronously
    /// </summary>
    private async Task<Accelerator?> CreateCpuFallbackAcceleratorAsync(CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                var cpuDevice = _context.GetCPUDevice(0);
                var cpuAccelerator = cpuDevice.CreateAccelerator(_context);
                _logger.LogDebug("Created CPU fallback accelerator: {Name}", cpuAccelerator.Name);
                return cpuAccelerator;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to create CPU fallback accelerator");
                throw new InvalidOperationException("No accelerators available for GPU bridge", ex);
            }
        }, cancellationToken).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU device manager");

        try
        {
            // Try async disposal first, fall back to sync
            var disposeTask = DisposeAsyncCore();
            if (!disposeTask.IsCompletedSuccessfully)
            {
                // Use timeout for disposal to prevent hanging
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                try
                {
                    disposeTask.AsTask().Wait(cts.Token);
                }
                catch (OperationCanceledException)
                {
                    _logger.LogWarning("Async disposal timed out, performing sync disposal");
                    DisposeSync();
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU device manager");
        }

        _disposed = true;
    }

    private void DisposeSync()
    {
        // Dispose all contexts
        foreach (var context in _contexts.Values)
        {
            try
            {
                context.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing context during sync disposal");
            }
        }
        _contexts.Clear();

        // Dispose all devices (which will dispose their accelerators)
        foreach (var device in _devices)
        {
            try
            {
                device.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing device during sync disposal");
            }
        }
        _devices.Clear();
    }

    private async ValueTask DisposeAsyncCore()
    {
        var disposeTasks = new List<Task>();

        // Dispose all contexts asynchronously
        foreach (var context in _contexts.Values)
        {
            disposeTasks.Add(Task.Run(() => context.Dispose()));
        }

        // Dispose all devices asynchronously
        foreach (var device in _devices)
        {
            disposeTasks.Add(Task.Run(() => device.Dispose()));
        }

        if (disposeTasks.Count > 0)
        {
            await Task.WhenAll(disposeTasks).ConfigureAwait(false);
        }

        _contexts.Clear();
        _devices.Clear();
    }
}