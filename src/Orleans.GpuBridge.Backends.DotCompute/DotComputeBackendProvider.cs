using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Backends.DotCompute.Configuration;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Orleans.GpuBridge.Backends.DotCompute.Kernels;
using Orleans.GpuBridge.Backends.DotCompute.Memory;
using Orleans.GpuBridge.Backends.DotCompute.Execution;

namespace Orleans.GpuBridge.Backends.DotCompute;

/// <summary>
/// DotCompute backend provider implementation for Orleans.GpuBridge.
/// Provides GPU acceleration via CUDA, OpenCL, Metal, and CPU with attribute-based kernel definition.
/// </summary>
public sealed class DotComputeBackendProvider : IGpuBackendProvider
{
    private readonly ILogger<DotComputeBackendProvider> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private readonly DotComputeOptions _options;

    private DotComputeDeviceManager? _deviceManager;
    private DotComputeKernelCompiler? _kernelCompiler;
    private DotComputeMemoryAllocator? _memoryAllocator;
    private DotComputeKernelExecutor? _kernelExecutor;
    private bool _isInitialized;
    private bool _disposed;

    /// <summary>
    /// Unique identifier for this backend provider.
    /// </summary>
    public string ProviderId => "DotCompute";

    /// <summary>
    /// Human-readable display name for this provider.
    /// </summary>
    public string DisplayName => "DotCompute Universal Backend";

    /// <summary>
    /// Provider version information.
    /// </summary>
    public Version Version => new(0, 2, 0);

    /// <summary>
    /// Backend capabilities supported by DotCompute.
    /// </summary>
    public BackendCapabilities Capabilities { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="DotComputeBackendProvider"/> class.
    /// </summary>
    /// <param name="logger">Logger for provider operations.</param>
    /// <param name="loggerFactory">Factory for creating component loggers.</param>
    /// <param name="options">DotCompute configuration options.</param>
    public DotComputeBackendProvider(
        ILogger<DotComputeBackendProvider> logger,
        ILoggerFactory loggerFactory,
        IOptions<DotComputeOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));

        // Initialize capabilities based on DotCompute features
        Capabilities = BackendCapabilities.CreateDotCompute();

        _logger.LogInformation(
            "DotCompute backend provider created (version {Version})",
            Version);
    }

    /// <summary>
    /// Initializes the DotCompute backend provider and discovers available compute devices.
    /// </summary>
    /// <param name="configuration">Backend configuration parameters.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the initialization operation.</returns>
    public async Task InitializeAsync(
        BackendConfiguration configuration,
        CancellationToken cancellationToken = default)
    {
        if (_isInitialized)
        {
            _logger.LogWarning("DotCompute provider already initialized");
            return;
        }

        _logger.LogInformation("Initializing DotCompute backend provider");

        try
        {
            // Initialize components in dependency order
            // 1. Device Manager (no dependencies)
            _deviceManager = new DotComputeDeviceManager(
                _loggerFactory.CreateLogger<DotComputeDeviceManager>());

            await _deviceManager.InitializeAsync(cancellationToken);

            // 2. Kernel Compiler (depends on device manager)
            _kernelCompiler = new DotComputeKernelCompiler(
                _deviceManager,
                _loggerFactory.CreateLogger<DotComputeKernelCompiler>());

            // 3. Memory Allocator (depends on device manager and configuration)
            _memoryAllocator = new DotComputeMemoryAllocator(
                _loggerFactory.CreateLogger<DotComputeMemoryAllocator>(),
                _deviceManager,
                configuration);

            // 4. Kernel Executor (depends on all other components)
            _kernelExecutor = new DotComputeKernelExecutor(
                _loggerFactory.CreateLogger<DotComputeKernelExecutor>(),
                _loggerFactory,
                _deviceManager,
                _memoryAllocator,
                _kernelCompiler);

            _isInitialized = true;

            var deviceCount = _deviceManager.GetDevices().Count;
            _logger.LogInformation(
                "DotCompute backend provider initialized successfully. Discovered {DeviceCount} device(s)",
                deviceCount);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize DotCompute backend provider");
            throw;
        }
    }

    /// <summary>
    /// Checks whether the DotCompute backend is available for use.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if at least one compute device is available; otherwise, false.</returns>
    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        if (!_isInitialized || _deviceManager == null)
        {
            return Task.FromResult(false);
        }

        // DotCompute is available if at least one device (including CPU) is available
        var deviceCount = _deviceManager.GetDevices().Count;
        var isAvailable = deviceCount > 0;

        _logger.LogDebug(
            "DotCompute availability check: {IsAvailable} ({DeviceCount} device(s))",
            isAvailable,
            deviceCount);

        return Task.FromResult(isAvailable);
    }

    /// <summary>
    /// Checks whether the DotCompute backend is available for use (synchronous version).
    /// </summary>
    /// <returns>True if at least one compute device is available; otherwise, false.</returns>
    public bool IsAvailable()
    {
        if (!_isInitialized || _deviceManager == null)
        {
            return false;
        }

        return _deviceManager.GetDevices().Count > 0;
    }

    /// <summary>
    /// Gets the device manager for enumerating and managing compute devices.
    /// </summary>
    /// <returns>The device manager instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the provider is not initialized.</exception>
    public IDeviceManager GetDeviceManager()
    {
        EnsureInitialized();
        return _deviceManager!;
    }

    /// <summary>
    /// Gets the kernel compiler for compiling compute kernels.
    /// </summary>
    /// <returns>The kernel compiler instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the provider is not initialized.</exception>
    public IKernelCompiler GetKernelCompiler()
    {
        EnsureInitialized();
        return _kernelCompiler!;
    }

    /// <summary>
    /// Gets the memory allocator for managing device memory.
    /// </summary>
    /// <returns>The memory allocator instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the provider is not initialized.</exception>
    public IMemoryAllocator GetMemoryAllocator()
    {
        EnsureInitialized();
        return _memoryAllocator!;
    }

    /// <summary>
    /// Gets the kernel executor for executing compiled kernels.
    /// </summary>
    /// <returns>The kernel executor instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the provider is not initialized.</exception>
    public IKernelExecutor GetKernelExecutor()
    {
        EnsureInitialized();
        return _kernelExecutor!;
    }

    /// <summary>
    /// Gets backend-specific metrics including device utilization, memory usage, and performance statistics.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A dictionary containing backend metrics.</returns>
    public async Task<IReadOnlyDictionary<string, object>> GetMetricsAsync(CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        var deviceCount = _deviceManager!.GetDevices().Count;
        var metrics = new Dictionary<string, object>
        {
            ["provider_id"] = ProviderId,
            ["provider_version"] = Version.ToString(),
            ["device_count"] = deviceCount,
            ["is_initialized"] = _isInitialized
        };

        // Aggregate device metrics if available
        try
        {
            var devices = _deviceManager.GetDevices();
            var deviceMetrics = new List<Dictionary<string, object>>();

            foreach (var device in devices)
            {
                try
                {
                    var deviceMetric = await _deviceManager.GetDeviceMetricsAsync(device, cancellationToken);
                    deviceMetrics.Add(new Dictionary<string, object>
                    {
                        ["device_id"] = device.DeviceId,
                        ["device_name"] = device.Name,
                        ["device_type"] = device.Type.ToString(),
                        ["gpu_utilization"] = deviceMetric.GpuUtilizationPercent,
                        ["memory_utilization"] = deviceMetric.MemoryUtilizationPercent,
                        ["used_memory_bytes"] = deviceMetric.UsedMemoryBytes,
                        ["temperature"] = deviceMetric.TemperatureCelsius,
                        ["power_watts"] = deviceMetric.PowerWatts
                    });
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to get metrics for device {DeviceId}", device.DeviceId);
                }
            }

            metrics["devices"] = deviceMetrics;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to aggregate device metrics");
        }

        return metrics;
    }

    /// <summary>
    /// Performs a health check on the backend and all compute devices.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Health check result indicating overall backend health.</returns>
    public async Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            return new HealthCheckResult(
                IsHealthy: false,
                Message: "DotCompute backend provider is not initialized");
        }

        if (_disposed)
        {
            return new HealthCheckResult(
                IsHealthy: false,
                Message: "DotCompute backend provider is disposed");
        }

        try
        {
            var devices = _deviceManager!.GetDevices();
            var healthyDevices = 0;
            var totalDevices = devices.Count;
            var diagnostics = new Dictionary<string, object>();

            foreach (var device in devices)
            {
                try
                {
                    var deviceHealth = await _deviceManager.GetDeviceHealthAsync(device.DeviceId, cancellationToken);
                    if (deviceHealth.Status == Abstractions.Enums.DeviceStatus.Available)
                    {
                        healthyDevices++;
                    }

                    diagnostics[$"device_{device.DeviceId}_status"] = deviceHealth.Status.ToString();
                    diagnostics[$"device_{device.DeviceId}_memory_usage"] = deviceHealth.MemoryUtilizationPercent;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Health check failed for device {DeviceId}", device.DeviceId);
                    diagnostics[$"device_{device.DeviceId}_error"] = ex.Message;
                }
            }

            diagnostics["healthy_devices"] = healthyDevices;
            diagnostics["total_devices"] = totalDevices;

            var isHealthy = healthyDevices > 0;
            var message = isHealthy
                ? $"DotCompute backend is healthy: {healthyDevices}/{totalDevices} devices available"
                : "DotCompute backend has no healthy devices available";

            return new HealthCheckResult(
                IsHealthy: isHealthy,
                Message: message,
                Diagnostics: diagnostics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Health check failed for DotCompute backend");
            return new HealthCheckResult(
                IsHealthy: false,
                Message: $"Health check failed: {ex.Message}",
                Diagnostics: new Dictionary<string, object> { ["exception"] = ex.ToString() });
        }
    }

    /// <summary>
    /// Creates a compute context for the specified device.
    /// </summary>
    /// <param name="deviceIndex">Index of the device to create a context for (default: 0).</param>
    /// <returns>A compute context object for the specified device.</returns>
    public async Task<object> CreateContext(int deviceIndex = 0)
    {
        EnsureInitialized();

        try
        {
            var device = _deviceManager!.GetDevice(deviceIndex);
            var context = await _deviceManager.CreateContextAsync(
                device,
                new Abstractions.Models.ContextOptions(),
                CancellationToken.None);

            _logger.LogInformation(
                "Created compute context for device {DeviceId} (index {DeviceIndex})",
                device.DeviceId,
                deviceIndex);

            return context;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create compute context for device index {DeviceIndex}", deviceIndex);
            throw;
        }
    }

    /// <summary>
    /// Disposes of resources used by the backend provider.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _logger.LogInformation("Disposing DotCompute backend provider");

        try
        {
            _deviceManager?.Dispose();
            _kernelCompiler?.Dispose();
            _memoryAllocator?.Dispose();
            _kernelExecutor?.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute backend provider");
        }
        finally
        {
            _disposed = true;
            _isInitialized = false;
        }
    }

    private void EnsureInitialized()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException(
                "DotCompute backend provider is not initialized. Call InitializeAsync first.");
        }

        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DotComputeBackendProvider));
        }
    }
}
