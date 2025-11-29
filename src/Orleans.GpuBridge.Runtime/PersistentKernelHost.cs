using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime.Persistent;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Host service for persistent GPU kernels with ring buffer I/O
/// </summary>
public sealed class PersistentKernelHost : IHostedService, IDisposable
{
    private readonly ILogger<PersistentKernelHost> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly PersistentKernelHostOptions _options;
    private readonly RingBufferManager _ringBufferManager;
    private readonly KernelLifecycleManager _lifecycleManager;
    private readonly KernelCatalog _kernelCatalog;
    private readonly Dictionary<string, PersistentKernelInstance> _runningKernels;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="PersistentKernelHost"/> class.
    /// </summary>
    /// <param name="logger">The logger for diagnostic output.</param>
    /// <param name="serviceProvider">The service provider for dependency resolution.</param>
    /// <param name="options">The host options configuration.</param>
    /// <param name="kernelCatalog">The kernel catalog for resolving kernels.</param>
    public PersistentKernelHost(
        ILogger<PersistentKernelHost> logger,
        IServiceProvider serviceProvider,
        IOptions<PersistentKernelHostOptions> options,
        KernelCatalog kernelCatalog)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _options = options.Value;
        _kernelCatalog = kernelCatalog;
        _runningKernels = new Dictionary<string, PersistentKernelInstance>();

        // Create managers
        _ringBufferManager = new RingBufferManager(
            serviceProvider.GetRequiredService<ILogger<RingBufferManager>>(),
            _options.DefaultRingBufferSize);

        _lifecycleManager = new KernelLifecycleManager(
            serviceProvider.GetRequiredService<ILogger<KernelLifecycleManager>>());
    }

    /// <inheritdoc/>
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting persistent kernel host with {Count} configured kernels",
            _options.KernelConfigurations.Count);

        foreach (var config in _options.KernelConfigurations)
        {
            try
            {
                await StartKernelAsync(config, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start kernel {KernelId}", config.KernelId);

                if (!_options.ContinueOnKernelFailure)
                {
                    throw;
                }
            }
        }

        _logger.LogInformation("Persistent kernel host started with {Count} running kernels",
            _runningKernels.Count);
    }

    private async Task StartKernelAsync(
        PersistentKernelConfiguration config,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting persistent kernel {KernelId}", config.KernelId);

        // Resolve kernel from catalog
        var kernel = await _kernelCatalog.ResolveAsync<byte[], byte[]>(
            new KernelId(config.KernelId),
            _serviceProvider);

        // Create ring buffer for this kernel
        var ringBuffer = _ringBufferManager.CreateBuffer(
            config.KernelId,
            config.RingBufferSize ?? _options.DefaultRingBufferSize);

        // Configure kernel options
        var kernelOptions = new PersistentKernelOptions
        {
            BatchSize = config.BatchSize ?? _options.DefaultBatchSize,
            MaxBatchWaitTime = config.MaxBatchWaitTime ?? _options.DefaultMaxBatchWaitTime,
            RestartOnError = config.RestartOnError ?? true,
            MaxRetries = config.MaxRetries ?? 3
        };

        // Start kernel instance
        var instance = await _lifecycleManager.StartKernelAsync(
            new KernelId(config.KernelId),
            kernel,
            kernelOptions,
            cancellationToken);

        _runningKernels[config.KernelId] = instance;

        _logger.LogInformation("Started persistent kernel {KernelId} successfully", config.KernelId);
    }

    /// <inheritdoc/>
    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Stopping persistent kernel host");

        var stopTasks = _runningKernels.Values
            .Select(k => _lifecycleManager.StopKernelAsync(k.InstanceId, cancellationToken))
            .ToList();

        await Task.WhenAll(stopTasks);

        _runningKernels.Clear();

        _logger.LogInformation("Persistent kernel host stopped");
    }

    /// <summary>
    /// Gets the status of all running kernels
    /// </summary>
    public Dictionary<string, KernelInstanceStatus> GetKernelStatuses()
    {
        return _lifecycleManager.GetAllStatuses();
    }

    /// <summary>
    /// Gets ring buffer statistics
    /// </summary>
    public Dictionary<string, RingBufferStats> GetBufferStatistics()
    {
        return _ringBufferManager.GetStatistics();
    }

    /// <summary>
    /// Restarts a specific kernel
    /// </summary>
    public async Task RestartKernelAsync(string kernelId, CancellationToken cancellationToken = default)
    {
        if (_runningKernels.TryGetValue(kernelId, out var instance))
        {
            await _lifecycleManager.RestartKernelAsync(instance.InstanceId, cancellationToken);
        }
        else
        {
            throw new InvalidOperationException($"Kernel {kernelId} is not running");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _lifecycleManager?.Dispose();
        _ringBufferManager?.Dispose();
    }
}

/// <summary>
/// Options for the persistent kernel host.
/// </summary>
public sealed class PersistentKernelHostOptions
{
    /// <summary>
    /// Gets the list of kernel configurations to start.
    /// </summary>
    public List<PersistentKernelConfiguration> KernelConfigurations { get; } = new();

    /// <summary>
    /// Gets or sets the default ring buffer size in bytes. Default is 16MB.
    /// </summary>
    public int DefaultRingBufferSize { get; set; } = 16 * 1024 * 1024; // 16MB

    /// <summary>
    /// Gets or sets the default batch size for processing. Default is 100.
    /// </summary>
    public int DefaultBatchSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the default maximum time to wait before flushing a partial batch. Default is 100ms.
    /// </summary>
    public TimeSpan DefaultMaxBatchWaitTime { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Gets or sets whether to continue starting other kernels if one fails. Default is true.
    /// </summary>
    public bool ContinueOnKernelFailure { get; set; } = true;

    /// <summary>
    /// Gets or sets whether health monitoring is enabled. Default is true.
    /// </summary>
    public bool EnableHealthMonitoring { get; set; } = true;

    /// <summary>
    /// Gets or sets the interval between health checks. Default is 30 seconds.
    /// </summary>
    public TimeSpan HealthCheckInterval { get; set; } = TimeSpan.FromSeconds(30);
}

/// <summary>
/// Configuration for a persistent kernel.
/// </summary>
public sealed class PersistentKernelConfiguration
{
    /// <summary>
    /// Gets or sets the kernel identifier.
    /// </summary>
    public string KernelId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ring buffer size in bytes. Uses host default if null.
    /// </summary>
    public int? RingBufferSize { get; set; }

    /// <summary>
    /// Gets or sets the batch size for processing. Uses host default if null.
    /// </summary>
    public int? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the maximum time to wait before flushing a partial batch. Uses host default if null.
    /// </summary>
    public TimeSpan? MaxBatchWaitTime { get; set; }

    /// <summary>
    /// Gets or sets whether to restart the kernel on error. Default is true if null.
    /// </summary>
    public bool? RestartOnError { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of restart retries. Default is 3 if null.
    /// </summary>
    public int? MaxRetries { get; set; }

    /// <summary>
    /// Gets or sets additional kernel parameters.
    /// </summary>
    public Dictionary<string, object>? Parameters { get; set; }
}
