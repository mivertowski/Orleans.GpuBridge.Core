using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions;
using DotCompute.Abstractions.Factories;
using DotCompute.Core.Compute;
using DotCompute.Runtime; // ✅ Unified namespace for AddDotComputeRuntime()
using DotCompute.Runtime.Configuration;
using DotCompute.Runtime.Factories;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using DotCompute.Abstractions.Health;
using DotCompute.Abstractions.Profiling;
using DotCompute.Abstractions.Recovery;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Models;

// Alias to resolve namespace conflict
using GpuBridgeDeviceMetrics = Orleans.GpuBridge.Abstractions.Models.DeviceMetrics;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// DotCompute device manager implementation with real DotCompute v0.5.1 API integration.
/// Uses IUnifiedAcceleratorFactory pattern for reliable device discovery.
/// </summary>
internal sealed class DotComputeDeviceManager : IDeviceManager
{
    private readonly ILogger<DotComputeDeviceManager> _logger;
    private readonly ConcurrentDictionary<string, DotComputeAcceleratorAdapter> _devices;
    private IUnifiedAcceleratorFactory? _factory;
    private IServiceProvider? _serviceProvider;
    private bool _initialized;
    private bool _disposed;

    public DotComputeDeviceManager(ILogger<DotComputeDeviceManager> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _devices = new ConcurrentDictionary<string, DotComputeAcceleratorAdapter>();
    }

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return;

        try
        {
            _logger.LogInformation("Initializing DotCompute device manager with unified DI API (v0.5.1)");

            // ✅ NEW API: Use Host.CreateApplicationBuilder with AddDotComputeRuntime()
            var hostBuilder = Host.CreateApplicationBuilder();

            // Add logging
            hostBuilder.Services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });

            // Configure DotCompute runtime options (optional, set before AddDotComputeRuntime)
            hostBuilder.Services.Configure<DotComputeRuntimeOptions>(options =>
            {
                // DotCompute 0.5.1+ handles WSL2 compatibility internally
                options.ValidateCapabilities = true;
                options.AcceleratorLifetime = global::DotCompute.Runtime.Configuration.ServiceLifetime.Transient;
            });

            // ✅ UNIFIED METHOD: Registers ALL services (factory, orchestrator, providers)
            hostBuilder.Services.AddDotComputeRuntime();

            var host = hostBuilder.Build();
            _serviceProvider = host.Services;
            _factory = _serviceProvider.GetRequiredService<IUnifiedAcceleratorFactory>();

            _logger.LogInformation("DotCompute IUnifiedAcceleratorFactory registered via AddDotComputeRuntime()");

            // ✅ NEW API: Discover devices using GetAvailableDevicesAsync
            await DiscoverDevicesAsync(cancellationToken).ConfigureAwait(false);

            _initialized = true;
            _logger.LogInformation(
                "DotCompute device manager initialized with {DeviceCount} real devices (CUDA: {CudaCount}, OpenCL: {OpenClCount}, CPU: {CpuCount})",
                _devices.Count,
                _devices.Values.Count(d => d.Type == DeviceType.CUDA),
                _devices.Values.Count(d => d.Type == DeviceType.OpenCL),
                _devices.Values.Count(d => d.Type == DeviceType.CPU));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize DotCompute device manager");
            throw;
        }
    }

    public IReadOnlyList<IComputeDevice> GetDevices()
    {
        EnsureInitialized();
        return _devices.Values.ToList();
    }

    public IComputeDevice GetDefaultDevice()
    {
        EnsureInitialized();

        // Prefer GPU devices over CPU
        var gpuDevice = _devices.Values.FirstOrDefault(d => d.Type != DeviceType.CPU);
        if (gpuDevice != null)
        {
            return gpuDevice;
        }

        // Fallback to any available device
        var firstDevice = _devices.Values.FirstOrDefault();
        if (firstDevice == null)
        {
            throw new InvalidOperationException("No compute devices available");
        }

        return firstDevice;
    }

    public IComputeDevice? GetDevice(string deviceId)
    {
        EnsureInitialized();
        _devices.TryGetValue(deviceId, out var device);
        return device;
    }

    public IEnumerable<IComputeDevice> GetDevicesByType(DeviceType deviceType)
    {
        EnsureInitialized();
        return _devices.Values.Where(d => d.Type == deviceType);
    }

    public Task<DeviceHealthInfo> GetDeviceHealthAsync(string deviceId, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (!_devices.TryGetValue(deviceId, out var device))
        {
            throw new ArgumentException($"Device not found: {deviceId}", nameof(deviceId));
        }

        try
        {
            // ✅ REAL API: Use adapter's GetMemoryInfo() method
            var memoryInfo = device.GetMemoryInfo();
            var memoryUsage = memoryInfo.UtilizationPercentage;

            // DotCompute v0.3.0-rc1: Temperature sensors not yet available
            // Using simulated temperature until sensor APIs are implemented
            var temperature = device.Type == DeviceType.GPU ? 45 : 0;

            var healthInfo = new DeviceHealthInfo
            {
                DeviceId = device.DeviceId,
                MemoryUtilizationPercent = memoryUsage,
                TemperatureCelsius = temperature,
                Status = device.IsHealthy ? DeviceStatus.Available : DeviceStatus.Error,
                ErrorCount = string.IsNullOrEmpty(device.LastError) ? 0 : 1,
                ConsecutiveFailures = device.IsHealthy ? 0 : 1
            };

            return Task.FromResult(healthInfo);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get health info for device {DeviceId}", deviceId);
            throw;
        }
    }

    /// <summary>
    /// Discovers available compute devices using DotCompute v0.4.0-rc2 GetAvailableDevicesAsync API
    /// </summary>
    private async Task DiscoverDevicesAsync(CancellationToken cancellationToken)
    {
        if (_factory == null)
            throw new InvalidOperationException("IUnifiedAcceleratorFactory not initialized");

        _logger.LogInformation("Starting DotCompute device discovery using IUnifiedAcceleratorFactory API");

        try
        {
            // ✅ NEW API: GetAvailableDevicesAsync returns device descriptors
            var deviceDescriptors = await _factory.GetAvailableDevicesAsync()
                .ConfigureAwait(false);

            _logger.LogDebug("Retrieved {Count} device descriptors", deviceDescriptors.Count);

            // Diagnostic: Log available methods on factory
            var factoryMethods = _factory.GetType().GetMethods()
                .Where(m => m.IsPublic && !m.IsSpecialName)
                .Select(m => $"{m.Name}({string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name))})")
                .ToList();
            _logger.LogDebug("Available factory methods: {Methods}", string.Join(", ", factoryMethods));

            // Diagnostic: Log device descriptor type
            if (deviceDescriptors.Count > 0)
            {
                var descType = deviceDescriptors[0].GetType();
                _logger.LogDebug("Device descriptor type: {Type}, Interfaces: {Interfaces}",
                    descType.FullName,
                    string.Join(", ", descType.GetInterfaces().Select(i => i.Name)));
            }

            var index = 0;
            foreach (var deviceDesc in deviceDescriptors)
            {
                cancellationToken.ThrowIfCancellationRequested();

                try
                {
                    // ✅ NEW API: Create accelerator from AcceleratorInfo
                    // Working pattern: await factory.CreateAsync(device)
                    var accelerator = await _factory.CreateAsync(deviceDesc)
                        .ConfigureAwait(false);

                    if (accelerator == null)
                    {
                        _logger.LogWarning(
                            "Factory returned null accelerator for device: {DeviceName} ({DeviceType}). Skipping.",
                            deviceDesc.Name,
                            deviceDesc.DeviceType);
                        continue;
                    }

                    // Create adapter to wrap IAccelerator as IComputeDevice
                    var adapter = new DotComputeAcceleratorAdapter(accelerator, index++, _logger);
                    _devices[adapter.Id] = adapter;

                    _logger.LogInformation(
                        "Discovered DotCompute device: {DeviceId} - {DeviceName} ({DeviceType}, {Architecture})",
                        adapter.Id,
                        adapter.Name,
                        adapter.Type,
                        adapter.Architecture);

                    _logger.LogDebug(
                        "Device details: ComputeUnits={ComputeUnits}, Memory={MemoryGB:F2}GB, WarpSize={WarpSize}",
                        adapter.ComputeUnits,
                        adapter.TotalMemoryBytes / (1024.0 * 1024.0 * 1024.0),
                        adapter.WarpSize);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex,
                        "Failed to process device: {DeviceName} ({DeviceType}). Skipping.",
                        deviceDesc.Name,
                        deviceDesc.DeviceType);
                }
            }

            _logger.LogInformation(
                "Device discovery complete. Found {DeviceCount} real device(s) across {DescriptorCount} descriptors",
                _devices.Count,
                deviceDescriptors.Count);

            if (_devices.Count == 0)
            {
                _logger.LogWarning(
                    "No devices discovered. This may be due to:" + Environment.NewLine +
                    "  - Missing CUDA/OpenCL drivers" + Environment.NewLine +
                    "  - WSL2 GPU passthrough not configured" + Environment.NewLine +
                    "  - No compatible compute devices available");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during device discovery");
            throw;
        }
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
            throw new InvalidOperationException("DotCompute device manager not initialized");
    }

    public IComputeDevice GetDevice(int index)
    {
        EnsureInitialized();
        var device = _devices.Values.FirstOrDefault(d => d.Index == index);
        if (device == null)
        {
            throw new ArgumentException($"Device not found at index: {index}", nameof(index));
        }
        return device;
    }

    public IComputeDevice SelectDevice(DeviceSelectionCriteria criteria)
    {
        EnsureInitialized();

        var availableDevices = _devices.Values.Where(d => d.IsHealthy);

        if (criteria.PreferredType.HasValue)
        {
            availableDevices = availableDevices.Where(d => d.Type == criteria.PreferredType.Value);
        }

        if (criteria.MinimumMemoryBytes > 0)
        {
            availableDevices = availableDevices.Where(d => d.AvailableMemoryBytes >= criteria.MinimumMemoryBytes);
        }

        if (criteria.MinComputeUnits > 0)
        {
            availableDevices = availableDevices.Where(d => d.ComputeUnits >= criteria.MinComputeUnits);
        }

        var selectedDevice = availableDevices.FirstOrDefault();
        if (selectedDevice == null)
        {
            throw new InvalidOperationException("No devices match the specified criteria");
        }

        return selectedDevice;
    }

    /// <summary>
    /// Creates a compute context for the specified device using DotCompute v0.5.1 implicit context management.
    /// </summary>
    /// <remarks>
    /// <para>DotCompute uses implicit context management via <see cref="IAccelerator.Context"/>.</para>
    /// <para>This method wraps the accelerator's context in an adapter for Orleans compatibility.</para>
    /// </remarks>
    public Task<IComputeContext> CreateContextAsync(IComputeDevice device, ContextOptions options, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (device == null)
            throw new ArgumentNullException(nameof(device));

        var adapter = device as DotComputeAcceleratorAdapter
            ?? throw new InvalidOperationException($"Device {device.DeviceId} is not a DotCompute device");

        var accelerator = adapter.Accelerator;

        // DotCompute v0.5.1 uses implicit context management via IAccelerator.Context property
        // The AcceleratorContext is automatically managed and provides execution context
        var nativeContext = accelerator.Context;

        _logger.LogDebug(
            "Created compute context for device {DeviceId} using DotCompute implicit context management",
            device.DeviceId);

        // Create a simple context wrapper for Orleans compatibility
        // DotCompute v0.5.1 uses implicit context via IAccelerator.Context
        var contextWrapper = new DotComputeImplicitContext(device, accelerator);

        return Task.FromResult<IComputeContext>(contextWrapper);
    }

    /// <summary>
    /// Gets device metrics using DotCompute v0.5.1 health monitoring APIs.
    /// </summary>
    /// <remarks>
    /// <para>Uses real DotCompute APIs for metrics collection:</para>
    /// <list type="bullet">
    ///   <item><see cref="IAccelerator.GetHealthSnapshotAsync"/> - overall health score</item>
    ///   <item><see cref="IAccelerator.GetSensorReadingsAsync"/> - temperature, power, fan</item>
    ///   <item><see cref="IAccelerator.GetProfilingSnapshotAsync"/> - GPU utilization</item>
    /// </list>
    /// <para>Falls back to memory-based metrics if profiling APIs are unavailable.</para>
    /// </remarks>
    public async Task<GpuBridgeDeviceMetrics> GetDeviceMetricsAsync(IComputeDevice device, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (device == null)
            throw new ArgumentNullException(nameof(device));

        var adapter = device as DotComputeAcceleratorAdapter
            ?? throw new InvalidOperationException($"Device {device.DeviceId} is not a DotCompute device");

        var accelerator = adapter.Accelerator;

        _logger.LogDebug("Getting metrics for device {DeviceId} using DotCompute APIs", device.DeviceId);

        try
        {
            // Gather metrics concurrently from multiple DotCompute APIs
            var healthTask = GetHealthSnapshotSafeAsync(accelerator, cancellationToken);
            var sensorsTask = GetSensorReadingsSafeAsync(accelerator, cancellationToken);
            var profilingTask = GetProfilingSnapshotSafeAsync(accelerator, cancellationToken);

            await Task.WhenAll(healthTask, sensorsTask, profilingTask);

            var healthSnapshot = await healthTask;
            var sensorReadings = await sensorsTask;
            var profilingSnapshot = await profilingTask;

            // Extract metrics from sensor readings
            var temperature = ExtractSensorValue(sensorReadings, "Temperature", "GPU_TEMP", "temp");
            var power = ExtractSensorValue(sensorReadings, "Power", "GPU_POWER", "power");
            var fanSpeed = ExtractSensorValue(sensorReadings, "FanSpeed", "FAN_SPEED", "fan");

            // Extract utilization from profiling snapshot
            var gpuUtilization = (float)(profilingSnapshot?.DeviceUtilizationPercent ?? 0.0);
            var memoryInfo = adapter.GetMemoryInfo();
            var memoryUtilization = (float)(profilingSnapshot?.MemoryStats?.MemoryUtilizationPercent ?? memoryInfo.UtilizationPercentage);

            // Calculate used memory
            var usedMemoryBytes = device.TotalMemoryBytes - memoryInfo.FreeMemoryBytes;

            // Extract kernel and transfer stats
            var kernelsExecuted = profilingSnapshot?.KernelStats?.TotalExecutions ?? 0;
            var bytesTransferred = (profilingSnapshot?.MemoryStats?.HostToDeviceBytes ?? 0) +
                                   (profilingSnapshot?.MemoryStats?.DeviceToHostBytes ?? 0);

            var metrics = new GpuBridgeDeviceMetrics
            {
                GpuUtilizationPercent = gpuUtilization,
                MemoryUtilizationPercent = memoryUtilization,
                UsedMemoryBytes = usedMemoryBytes,
                TemperatureCelsius = temperature,
                PowerWatts = power,
                FanSpeedPercent = fanSpeed,
                KernelsExecuted = kernelsExecuted,
                BytesTransferred = bytesTransferred,
                Uptime = TimeSpan.FromMilliseconds(Environment.TickCount64)
            };

            _logger.LogDebug(
                "Device {DeviceId} metrics: GPU={GpuUtil:F1}%, Mem={MemUtil:F1}%, Temp={Temp:F0}°C, Power={Power:F0}W",
                device.DeviceId, gpuUtilization, memoryUtilization, temperature, power);

            return metrics;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Failed to get full metrics for device {DeviceId}, using fallback", device.DeviceId);
            return GetFallbackMetrics(device, adapter);
        }
    }

    /// <summary>
    /// Safely gets health snapshot with exception handling.
    /// </summary>
    private async Task<DeviceHealthSnapshot?> GetHealthSnapshotSafeAsync(IAccelerator accelerator, CancellationToken ct)
    {
        try
        {
            return await accelerator.GetHealthSnapshotAsync(ct);
        }
        catch (NotImplementedException)
        {
            _logger.LogDebug("GetHealthSnapshotAsync not implemented for {AcceleratorType}", accelerator.Type);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to get health snapshot");
            return null;
        }
    }

    /// <summary>
    /// Safely gets sensor readings with exception handling.
    /// </summary>
    private async Task<IReadOnlyList<SensorReading>?> GetSensorReadingsSafeAsync(IAccelerator accelerator, CancellationToken ct)
    {
        try
        {
            return await accelerator.GetSensorReadingsAsync(ct);
        }
        catch (NotImplementedException)
        {
            _logger.LogDebug("GetSensorReadingsAsync not implemented for {AcceleratorType}", accelerator.Type);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to get sensor readings");
            return null;
        }
    }

    /// <summary>
    /// Safely gets profiling snapshot with exception handling.
    /// </summary>
    private async Task<ProfilingSnapshot?> GetProfilingSnapshotSafeAsync(IAccelerator accelerator, CancellationToken ct)
    {
        try
        {
            return await accelerator.GetProfilingSnapshotAsync(ct);
        }
        catch (NotImplementedException)
        {
            _logger.LogDebug("GetProfilingSnapshotAsync not implemented for {AcceleratorType}", accelerator.Type);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to get profiling snapshot");
            return null;
        }
    }

    /// <summary>
    /// Extracts a sensor value by searching for matching sensor names or sensor types.
    /// </summary>
    private static float ExtractSensorValue(IReadOnlyList<SensorReading>? readings, params string[] sensorNames)
    {
        if (readings == null || readings.Count == 0)
            return 0f;

        // First try to match by name
        foreach (var name in sensorNames)
        {
            var reading = readings.FirstOrDefault(r =>
                r.IsAvailable && r.Name?.Contains(name, StringComparison.OrdinalIgnoreCase) == true);

            if (reading != null)
                return (float)reading.Value;
        }

        // Try to match by sensor type based on common patterns
        foreach (var name in sensorNames)
        {
            var sensorType = name.ToUpperInvariant() switch
            {
                "TEMPERATURE" or "GPU_TEMP" or "TEMP" => SensorType.Temperature,
                "POWER" or "GPU_POWER" => SensorType.PowerDraw,
                "FANSPEED" or "FAN_SPEED" or "FAN" => SensorType.FanSpeed,
                _ => (SensorType?)null
            };

            if (sensorType.HasValue)
            {
                var reading = readings.FirstOrDefault(r =>
                    r.IsAvailable && r.SensorType == sensorType.Value);

                if (reading != null)
                    return (float)reading.Value;
            }
        }

        return 0f;
    }

    /// <summary>
    /// Gets fallback metrics when DotCompute APIs are unavailable.
    /// </summary>
    private GpuBridgeDeviceMetrics GetFallbackMetrics(IComputeDevice device, DotComputeAcceleratorAdapter adapter)
    {
        var memoryInfo = adapter.GetMemoryInfo();
        var usedMemoryBytes = device.TotalMemoryBytes - memoryInfo.FreeMemoryBytes;

        return new GpuBridgeDeviceMetrics
        {
            GpuUtilizationPercent = 0f, // Unknown
            MemoryUtilizationPercent = (float)memoryInfo.UtilizationPercentage,
            UsedMemoryBytes = usedMemoryBytes,
            TemperatureCelsius = 0f, // Unknown
            PowerWatts = 0f, // Unknown
            FanSpeedPercent = 0f, // Unknown
            KernelsExecuted = 0,
            BytesTransferred = 0,
            Uptime = TimeSpan.FromMilliseconds(Environment.TickCount64)
        };
    }

    /// <summary>
    /// Resets the device using DotCompute v0.5.1 <see cref="IAccelerator.ResetAsync"/> API.
    /// </summary>
    /// <remarks>
    /// <para>Uses DotCompute's 5-tier reset strategy based on device state:</para>
    /// <list type="bullet">
    ///   <item><b>Soft</b>: Flush queues only (1-10ms)</item>
    ///   <item><b>Context</b>: Clear caches (10-50ms)</item>
    ///   <item><b>Hard</b>: Clear memory (50-200ms)</item>
    ///   <item><b>Full</b>: Complete reinitialization (200-1000ms)</item>
    ///   <item><b>ErrorRecovery</b>: Aggressive reset for error states</item>
    /// </list>
    /// <para>Automatically selects reset strategy based on device health.</para>
    /// </remarks>
    public async Task ResetDeviceAsync(IComputeDevice device, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (device == null)
            throw new ArgumentNullException(nameof(device));

        var adapter = device as DotComputeAcceleratorAdapter
            ?? throw new InvalidOperationException($"Device {device.DeviceId} is not a DotCompute device");

        var accelerator = adapter.Accelerator;

        _logger.LogWarning("Resetting device {DeviceId}", device.DeviceId);

        try
        {
            // Determine reset strategy based on device health
            var resetOptions = await DetermineResetStrategyAsync(accelerator, cancellationToken);

            _logger.LogDebug(
                "Using {ResetType} reset strategy for device {DeviceId}",
                resetOptions.ResetType, device.DeviceId);

            // Execute reset via DotCompute API
            var result = await accelerator.ResetAsync(resetOptions, cancellationToken);

            if (result.Success)
            {
                _logger.LogInformation(
                    "Device {DeviceId} reset completed successfully in {Duration}ms (Type: {ResetType})",
                    device.DeviceId, result.Duration.TotalMilliseconds, resetOptions.ResetType);

                // Update adapter state after successful reset
                adapter.SetHealthStatus(true);
            }
            else
            {
                _logger.LogError(
                    "Device {DeviceId} reset failed: {Error}",
                    device.DeviceId, result.ErrorMessage);

                throw new InvalidOperationException(
                    $"Failed to reset device {device.DeviceId}: {result.ErrorMessage}");
            }
        }
        catch (NotImplementedException)
        {
            // DotCompute backend doesn't implement ResetAsync - use fallback
            _logger.LogWarning(
                "ResetAsync not implemented for {AcceleratorType}, using synchronization fallback",
                accelerator.Type);

            await FallbackResetAsync(accelerator, device.DeviceId, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException && ex is not InvalidOperationException)
        {
            _logger.LogError(ex, "Unexpected error resetting device {DeviceId}", device.DeviceId);
            throw;
        }
    }

    /// <summary>
    /// Determines the appropriate reset strategy based on device health.
    /// </summary>
    private async Task<ResetOptions> DetermineResetStrategyAsync(IAccelerator accelerator, CancellationToken ct)
    {
        try
        {
            var health = await accelerator.GetHealthSnapshotAsync(ct);

            // Choose reset strategy based on health score
            return health.HealthScore switch
            {
                >= 0.9f => ResetOptions.Soft,           // Healthy - minimal reset
                >= 0.7f => ResetOptions.Context,        // Minor issues - clear caches
                >= 0.5f => ResetOptions.Hard,           // Degraded - clear memory
                >= 0.3f => ResetOptions.Full,           // Severely degraded - full reset
                _ => ResetOptions.ErrorRecovery         // Critical - aggressive recovery
            };
        }
        catch
        {
            // If health check fails, use conservative approach
            return ResetOptions.Hard;
        }
    }

    /// <summary>
    /// Fallback reset using synchronization when ResetAsync is not implemented.
    /// </summary>
    private async Task FallbackResetAsync(IAccelerator accelerator, string deviceId, CancellationToken ct)
    {
        _logger.LogDebug("Executing fallback reset for device {DeviceId}", deviceId);

        // Step 1: Synchronize to complete pending operations
        _logger.LogDebug("Synchronizing device {DeviceId}", deviceId);
        await accelerator.SynchronizeAsync(ct);

        // Step 2: Wait for GPU to settle
        await Task.Delay(50, ct);

        _logger.LogInformation("Device {DeviceId} fallback reset completed", deviceId);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute device manager");

        try
        {
            // Try async disposal with timeout
            var disposeTask = DisposeAsyncCore();
            if (!disposeTask.IsCompletedSuccessfully)
            {
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                try
                {
                    disposeTask.AsTask().Wait(cts.Token);
                }
                catch (OperationCanceledException)
                {
                    _logger.LogWarning("Async disposal timed out, performing sync disposal");
                    DisposeSynchronously();
                }
            }

            // Dispose service provider
            if (_serviceProvider is IDisposable disposableProvider)
            {
                disposableProvider.Dispose();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute device manager");
        }

        _disposed = true;
    }

    private void DisposeSynchronously()
    {
        foreach (var device in _devices.Values)
        {
            try
            {
                device.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing device {DeviceId}", device.DeviceId);
            }
        }
        _devices.Clear();
    }

    private async ValueTask DisposeAsyncCore()
    {
        var disposeTasks = _devices.Values.Select(device =>
            Task.Run(() =>
            {
                try
                {
                    device.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error disposing device {DeviceId}", device.DeviceId);
                }
            })
        ).ToArray();

        if (disposeTasks.Length > 0)
        {
            await Task.WhenAll(disposeTasks).ConfigureAwait(false);
        }

        _devices.Clear();
    }
}
