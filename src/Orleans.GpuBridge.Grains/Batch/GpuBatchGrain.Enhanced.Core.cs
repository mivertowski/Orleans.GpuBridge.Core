using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Concurrency;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

namespace Orleans.GpuBridge.Grains.Batch;

/// <summary>
/// Production-grade GPU batch processing grain with DotCompute integration
/// </summary>
/// <remarks>
/// Phase 2, Day 6-7 Enhancement:
/// - Real GPU execution via DotCompute backend
/// - Intelligent batch size optimization based on GPU memory
/// - Comprehensive performance metrics and profiling
/// - Multi-GPU support with device selection
/// - Graceful CPU fallback when GPU unavailable
///
/// Features:
/// - [StatelessWorker(1)]: One instance per silo for optimal GPU utilization
/// - [Reentrant]: Concurrent batch processing with semaphore control
/// - Adaptive batch sizing: Automatically splits large batches to fit GPU memory
/// - Performance tracking: Detailed metrics for monitoring and optimization
/// </remarks>
[StatelessWorker(1)] // One per silo for better GPU utilization
[Reentrant] // Allow concurrent calls
public sealed partial class GpuBatchGrainEnhanced<TIn, TOut> : Grain, IGpuBatchGrain<TIn, TOut>
    where TIn : unmanaged // Requires unmanaged types for GPU memory transfer
    where TOut : unmanaged
{
    private readonly ILogger<GpuBatchGrainEnhanced<TIn, TOut>> _logger;
    private readonly SemaphoreSlim _concurrencyLimit;

    // DotCompute backend integration
    private IGpuBackendProvider? _backendProvider;
    private IDeviceManager? _deviceManager;
    private IKernelExecutor? _kernelExecutor;
    private IMemoryAllocator? _memoryAllocator;
    private IKernelCompiler? _kernelCompiler;
    private CompiledKernel? _compiledKernel = null;

    // Kernel identity and configuration
    private KernelId _kernelId = default!;
    private IComputeDevice? _primaryDevice;

    // Performance tracking
    private long _totalItemsProcessed;
    private long _totalBatchesProcessed;
    private TimeSpan _totalGpuExecutionTime;
    private readonly Stopwatch _lifetimeStopwatch;

    /// <summary>
    /// Configuration for batch size optimization
    /// </summary>
    private const double GPU_MEMORY_UTILIZATION_TARGET = 0.8; // Use 80% of available memory
    private const int MIN_BATCH_SIZE = 256; // Minimum items per batch
    private const int DEFAULT_MAX_CONCURRENCY = 4; // Concurrent batch executions

    public GpuBatchGrainEnhanced(ILogger<GpuBatchGrainEnhanced<TIn, TOut>> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _concurrencyLimit = new SemaphoreSlim(
            DEFAULT_MAX_CONCURRENCY,
            DEFAULT_MAX_CONCURRENCY);
        _lifetimeStopwatch = Stopwatch.StartNew();
    }

    public override async Task OnActivateAsync(CancellationToken ct)
    {
        _kernelId = KernelId.Parse(this.GetPrimaryKeyString());

        _logger.LogInformation(
            "Activating GPU batch grain for kernel {KernelId}",
            _kernelId);

        try
        {
            // Initialize DotCompute backend
            await InitializeBackendAsync(ct).ConfigureAwait(false);

            // Compile kernel if backend is available
            if (_backendProvider != null && _kernelCompiler != null)
            {
                await CompileKernelAsync(ct).ConfigureAwait(false);
            }
            else
            {
                _logger.LogWarning(
                    "No GPU backend available for kernel {KernelId}, will use CPU fallback",
                    _kernelId);
            }

            _logger.LogInformation(
                "Activated GPU batch grain for kernel {KernelId} on device {DeviceType}",
                _kernelId,
                _primaryDevice?.Type ?? DeviceType.CPU);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to initialize GPU backend for kernel {KernelId}, falling back to CPU",
                _kernelId);
            // Continue with CPU fallback
        }

        await base.OnActivateAsync(ct).ConfigureAwait(false);
    }
}
