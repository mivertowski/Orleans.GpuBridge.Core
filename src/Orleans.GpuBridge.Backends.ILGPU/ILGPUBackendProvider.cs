using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;
using Orleans.GpuBridge.Backends.ILGPU.Kernels;
using Orleans.GpuBridge.Backends.ILGPU.Memory;
using Orleans.GpuBridge.Backends.ILGPU.Execution;

namespace Orleans.GpuBridge.Backends.ILGPU;

/// <summary>
/// ILGPU backend provider for Orleans GPU Bridge
/// </summary>
public sealed class ILGPUBackendProvider : IGpuBackendProvider
{
    private readonly ILogger<ILGPUBackendProvider> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private Context? _context;
    private ILGPUDeviceManager? _deviceManager;
    private ILGPUKernelCompiler? _kernelCompiler;
    private ILGPUMemoryAllocator? _memoryAllocator;
    private ILGPUKernelExecutor? _kernelExecutor;
    private BackendConfiguration _configuration = new();
    private bool _initialized;
    private bool _disposed;

    public string ProviderId => "ILGPU";
    public string DisplayName => "ILGPU Backend Provider";
    public Version Version => new(1, 0, 0);
    public BackendCapabilities Capabilities => BackendCapabilities.CreateILGPU();

    public ILGPUBackendProvider(ILogger<ILGPUBackendProvider> logger, ILoggerFactory loggerFactory)
    {
        _logger = logger;
        _loggerFactory = loggerFactory;
    }

    public async Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default)
    {
        if (_initialized)
        {
            _logger.LogWarning("ILGPU backend provider already initialized");
            return;
        }

        _logger.LogInformation("Initializing ILGPU backend provider");

        try
        {
            _configuration = configuration;

            // Initialize ILGPU context
            // Note: ContextFlags were removed in ILGPU 1.5+
            // Options are now set via builder pattern
            _context = Context.CreateDefault();

            // Initialize components
            _deviceManager = new ILGPUDeviceManager(_loggerFactory.CreateLogger<ILGPUDeviceManager>(), _context);
            await _deviceManager.InitializeAsync(cancellationToken).ConfigureAwait(false);

            _kernelCompiler = new ILGPUKernelCompiler(
                _loggerFactory.CreateLogger<ILGPUKernelCompiler>(), 
                _context,
                _deviceManager);

            _memoryAllocator = new ILGPUMemoryAllocator(
                _loggerFactory.CreateLogger<ILGPUMemoryAllocator>(),
                _deviceManager,
                configuration);

            _kernelExecutor = new ILGPUKernelExecutor(
                _loggerFactory.CreateLogger<ILGPUKernelExecutor>(),
                _deviceManager,
                _memoryAllocator);

            _initialized = true;

            _logger.LogInformation(
                "ILGPU backend provider initialized with {DeviceCount} devices",
                _deviceManager.GetDevices().Count);

            // Log available devices
            foreach (var device in _deviceManager.GetDevices())
            {
                _logger.LogInformation(
                    "ILGPU Device: {DeviceName} ({DeviceType}) - {MemoryMB} MB",
                    device.Name,
                    device.Type,
                    device.TotalMemoryBytes / (1024 * 1024));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize ILGPU backend provider");
            await DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized)
        {
            return Task.FromResult(false);
        }

        try
        {
            // Check if we have any usable accelerators
            var devices = _deviceManager?.GetDevices() ?? Array.Empty<IComputeDevice>();
            var hasUsableDevice = devices.Any(d => d.Type != DeviceType.CPU || d.GetStatus() == DeviceStatus.Available);
            
            return Task.FromResult(hasUsableDevice);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error checking ILGPU backend availability");
            return Task.FromResult(false);
        }
    }
    
    public bool IsAvailable()
    {
        if (!_initialized)
        {
            return false;
        }

        try
        {
            // Check if we have any usable accelerators
            var devices = _deviceManager?.GetDevices() ?? Array.Empty<IComputeDevice>();
            return devices.Any(d => d.Type != DeviceType.CPU || d.GetStatus() == DeviceStatus.Available);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error checking ILGPU backend availability");
            return false;
        }
    }

    public IDeviceManager GetDeviceManager()
    {
        EnsureInitialized();
        return _deviceManager!;
    }

    public IKernelCompiler GetKernelCompiler()
    {
        EnsureInitialized();
        return _kernelCompiler!;
    }

    public IMemoryAllocator GetMemoryAllocator()
    {
        EnsureInitialized();
        return _memoryAllocator!;
    }

    public IKernelExecutor GetKernelExecutor()
    {
        EnsureInitialized();
        return _kernelExecutor!;
    }

    public async Task<IReadOnlyDictionary<string, object>> GetMetricsAsync(CancellationToken cancellationToken = default)
    {
        var metrics = new Dictionary<string, object>
        {
            ["provider"] = ProviderId,
            ["version"] = Version.ToString(),
            ["initialized"] = _initialized
        };

        if (_initialized && _deviceManager != null)
        {
            var devices = _deviceManager.GetDevices();
            metrics["device_count"] = devices.Count;
            metrics["gpu_devices"] = devices.Count(d => d.Type != DeviceType.CPU);
            metrics["cpu_devices"] = devices.Count(d => d.Type == DeviceType.CPU);

            // Get memory metrics
            if (_memoryAllocator != null)
            {
                var memStats = _memoryAllocator.GetPoolStatistics();
                metrics["total_memory_allocated"] = memStats.TotalBytesAllocated;
                metrics["total_memory_in_use"] = memStats.TotalBytesInUse;
                metrics["fragmentation_percent"] = memStats.FragmentationPercent;
            }

            // Get device-specific metrics
            for (int i = 0; i < devices.Count && i < 4; i++) // Limit to first 4 devices
            {
                try
                {
                    var device = devices[i];
                    var deviceMetrics = await _deviceManager.GetDeviceMetricsAsync(device, cancellationToken).ConfigureAwait(false);
                    metrics[$"device_{i}_utilization"] = deviceMetrics.GpuUtilizationPercent;
                    metrics[$"device_{i}_memory_used"] = deviceMetrics.UsedMemoryBytes;
                    metrics[$"device_{i}_temperature"] = deviceMetrics.TemperatureCelsius;
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Failed to get metrics for device {DeviceIndex}", i);
                }
            }
        }

        return metrics;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized)
        {
            return new HealthCheckResult(false, "Provider not initialized");
        }

        if (_context == null || _deviceManager == null)
        {
            return new HealthCheckResult(false, "Core components not available");
        }

        try
        {
            // Check if context is still valid
            if (_context.IsDisposed)
            {
                return new HealthCheckResult(false, "ILGPU context is disposed");
            }

            // Check if we have available devices
            var devices = _deviceManager.GetDevices();
            if (devices.Count == 0)
            {
                return new HealthCheckResult(false, "No compute devices available");
            }

            // Check device status
            var availableDevices = devices.Count(d => d.GetStatus() == DeviceStatus.Available);
            if (availableDevices == 0)
            {
                return new HealthCheckResult(false, "No devices are currently available");
            }

            // Try a simple kernel compilation test (optional)
            if (_configuration.CustomSettings?.ContainsKey("health_check_kernel_test") == true)
            {
                await PerformKernelHealthCheckAsync(cancellationToken).ConfigureAwait(false);
            }

            var diagnostics = new Dictionary<string, object>
            {
                ["context_disposed"] = _context.IsDisposed,
                ["total_devices"] = devices.Count,
                ["available_devices"] = availableDevices,
                ["memory_pool_mb"] = _configuration.MaxMemoryPoolSizeMB
            };

            return new HealthCheckResult(true, "ILGPU backend is healthy", diagnostics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ILGPU backend health check failed");
            return new HealthCheckResult(false, $"Health check failed: {ex.Message}");
        }
    }

    private async Task PerformKernelHealthCheckAsync(CancellationToken cancellationToken)
    {
        // Simple test to verify kernel compilation works
        if (_kernelCompiler != null)
        {
            try
            {
                var testKernel = await _kernelCompiler.CompileFromSourceAsync(
                    "void TestKernel(Index1D index, ArrayView<int> data) { data[index] = index.X; }",
                    "TestKernel",
                    Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage.CSharp,
                    new KernelCompilationOptions(Orleans.GpuBridge.Abstractions.Enums.Compilation.OptimizationLevel.O1),
                    cancellationToken).ConfigureAwait(false);

                testKernel.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Kernel compilation health check failed");
                throw new InvalidOperationException("Kernel compilation test failed", ex);
            }
        }
    }

    public async Task<object> CreateContext(int deviceIndex = 0)
    {
        EnsureInitialized();
        
        if (_deviceManager == null)
            throw new InvalidOperationException("Device manager not initialized");
            
        var device = _deviceManager.GetDevice(deviceIndex);
        if (device == null)
            throw new ArgumentException($"Device at index {deviceIndex} not found", nameof(deviceIndex));
            
        var context = await _deviceManager.CreateContextAsync(device, new ContextOptions(), CancellationToken.None);
        return context;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("ILGPU backend provider not initialized");
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogInformation("Disposing ILGPU backend provider");

        try
        {
            _kernelExecutor?.Dispose();
            _memoryAllocator?.Dispose();
            _kernelCompiler?.Dispose();
            _deviceManager?.Dispose();
            _context?.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU backend provider components");
        }

        _disposed = true;
    }

    private ValueTask DisposeAsync()
    {
        if (_disposed)
            return default;

        _logger.LogInformation("Disposing ILGPU backend provider (async)");

        try
        {
            if (_kernelExecutor != null)
                _kernelExecutor.Dispose();

            if (_memoryAllocator != null)
                _memoryAllocator.Dispose();

            if (_kernelCompiler != null)
                _kernelCompiler.Dispose();

            if (_deviceManager != null)
                _deviceManager.Dispose();

            if (_context != null)
                _context.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU backend provider components");
        }

        _disposed = true;
        return default;
    }
}