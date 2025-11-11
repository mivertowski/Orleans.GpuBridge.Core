using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU fallback backend provider
/// </summary>
internal sealed class CpuFallbackProvider : IGpuBackendProvider
{
    private readonly ILogger<CpuFallbackProvider> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private CpuDeviceManager? _deviceManager;
    private CpuKernelCompiler? _kernelCompiler;
    private CpuMemoryAllocator? _memoryAllocator;
    private CpuKernelExecutor? _kernelExecutor;
    private bool _initialized;

    public string ProviderId => "CPU";
    public string DisplayName => "CPU Fallback Provider";
    public Version Version => new(1, 0, 0);
    public BackendCapabilities Capabilities => BackendCapabilities.CreateCpuFallback();

    public CpuFallbackProvider(ILogger<CpuFallbackProvider> logger, ILoggerFactory loggerFactory)
    {
        _logger = logger;
        _loggerFactory = loggerFactory;
    }

    public Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return Task.CompletedTask;

        _logger.LogInformation("Initializing CPU fallback backend provider");

        _deviceManager = new CpuDeviceManager(_loggerFactory.CreateLogger<CpuDeviceManager>());
        _kernelCompiler = new CpuKernelCompiler(_loggerFactory.CreateLogger<CpuKernelCompiler>());
        _memoryAllocator = new CpuMemoryAllocator(_loggerFactory.CreateLogger<CpuMemoryAllocator>());
        _kernelExecutor = new CpuKernelExecutor(_loggerFactory.CreateLogger<CpuKernelExecutor>());

        _initialized = true;
        return Task.CompletedTask;
    }

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true); // CPU is always available
    }

    public bool IsAvailable()
    {
        return true; // CPU is always available
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

    public Task<IReadOnlyDictionary<string, object>> GetMetricsAsync(CancellationToken cancellationToken = default)
    {
        var metrics = new Dictionary<string, object>
        {
            ["provider"] = ProviderId,
            ["cpu_cores"] = Environment.ProcessorCount,
            ["memory_mb"] = GC.GetTotalMemory(false) / (1024 * 1024)
        };

        return Task.FromResult<IReadOnlyDictionary<string, object>>(metrics);
    }

    public Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new HealthCheckResult(
            IsHealthy: true,
            Message: "CPU fallback provider is healthy"));
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
            throw new InvalidOperationException("Provider not initialized");
        }
    }

    public void Dispose()
    {
        _deviceManager?.Dispose();
        _kernelCompiler?.Dispose();
        _memoryAllocator?.Dispose();
        _kernelExecutor?.Dispose();
    }
}
