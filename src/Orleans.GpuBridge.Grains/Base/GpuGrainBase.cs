using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Grains.Base;

/// <summary>
/// Base class for Orleans grains with GPU acceleration support.
/// Provides Orleans-integrated kernel lifecycle management and telemetry.
/// </summary>
/// <typeparam name="TState">Grain state type</typeparam>
/// <remarks>
/// **Hybrid Layered Architecture:**
/// This facade provides Orleans integration for standard use cases:
/// - Automatic kernel lifecycle management (OnActivate/OnDeactivate)
/// - Grain placement awareness (GPU device affinity)
/// - Telemetry and monitoring hooks
/// - CPU fallback on GPU failure
///
/// For advanced users needing full DotCompute power, use DotCompute directly.
/// </remarks>
public abstract class GpuGrainBase<TState> : Grain, IGrainBase
    where TState : class, new()
{
    private readonly ILogger _logger;
    private readonly IGrainContext _grainContext;
    private bool _isInitialized;
    private bool _isDisposed;

    /// <summary>
    /// Grain state (Orleans-managed)
    /// </summary>
    protected TState State { get; private set; } = new();

    /// <summary>
    /// GPU device ID this grain is bound to (for multi-GPU systems)
    /// </summary>
    protected int GpuDeviceId { get; private set; } = 0;

    /// <summary>
    /// Whether GPU acceleration is currently available
    /// </summary>
    protected bool IsGpuAvailable { get; private set; }

    protected GpuGrainBase(IGrainContext grainContext, ILogger logger)
    {
        _grainContext = grainContext ?? throw new ArgumentNullException(nameof(grainContext));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    IGrainContext IGrainBase.GrainContext => _grainContext;

    /// <summary>
    /// Called when the grain is activated.
    /// Override to configure GPU kernels and resources.
    /// </summary>
    public override async Task OnActivateAsync(CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation(
                "Activating GPU grain {GrainType} with ID {GrainId}",
                GetType().Name,
                this.GetPrimaryKeyString());

            // Determine GPU device placement
            GpuDeviceId = await DetermineGpuDevicePlacementAsync(cancellationToken);

            // Check GPU availability
            IsGpuAvailable = await CheckGpuAvailabilityAsync(cancellationToken);

            if (!IsGpuAvailable)
            {
                _logger.LogWarning(
                    "GPU not available for grain {GrainType}, falling back to CPU execution",
                    GetType().Name);
            }

            // Initialize GPU resources
            await ConfigureGpuResourcesAsync(cancellationToken);

            _isInitialized = true;

            _logger.LogInformation(
                "GPU grain {GrainType} activated successfully on device {DeviceId} (GPU: {IsGpu})",
                GetType().Name,
                GpuDeviceId,
                IsGpuAvailable);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Failed to activate GPU grain {GrainType}",
                GetType().Name);
            throw;
        }
    }

    /// <summary>
    /// Called when the grain is deactivated.
    /// Cleans up GPU resources.
    /// </summary>
    public override async Task OnDeactivateAsync(DeactivationReason reason, CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation(
                "Deactivating GPU grain {GrainType} (Reason: {Reason})",
                GetType().Name,
                reason.ReasonCode);

            if (_isInitialized && !_isDisposed)
            {
                await CleanupGpuResourcesAsync(cancellationToken);
                _isDisposed = true;
            }

            _logger.LogInformation(
                "GPU grain {GrainType} deactivated successfully",
                GetType().Name);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error during GPU grain {GrainType} deactivation",
                GetType().Name);
            // Don't rethrow - deactivation should complete
        }
    }

    /// <summary>
    /// Determines which GPU device this grain should be placed on.
    /// Override for custom placement strategies (e.g., load balancing, affinity).
    /// </summary>
    protected virtual Task<int> DetermineGpuDevicePlacementAsync(CancellationToken cancellationToken)
    {
        // Default: Use device 0 (primary GPU)
        // Override for multi-GPU load balancing
        return Task.FromResult(0);
    }

    /// <summary>
    /// Checks if GPU execution is available.
    /// </summary>
    protected virtual Task<bool> CheckGpuAvailabilityAsync(CancellationToken cancellationToken)
    {
        // TODO: Query GpuBridgeProviderSelector for available GPU devices
        // For now, assume CPU fallback
        return Task.FromResult(false);
    }

    /// <summary>
    /// Configure GPU resources and kernels for this grain.
    /// Override to initialize kernels using IGpuKernel abstraction.
    /// </summary>
    /// <example>
    /// <code>
    /// protected override async Task ConfigureGpuResourcesAsync(CancellationToken cancellationToken)
    /// {
    ///     _myKernel = kernelFactory.CreateKernel&lt;float[], float[]&gt;("VectorAdd");
    ///     await _myKernel.InitializeAsync(cancellationToken);
    ///     await _myKernel.WarmupAsync(cancellationToken);
    /// }
    /// </code>
    /// </example>
    protected virtual Task ConfigureGpuResourcesAsync(CancellationToken cancellationToken)
    {
        // Default: No GPU resources to configure
        return Task.CompletedTask;
    }

    /// <summary>
    /// Cleanup GPU resources on deactivation.
    /// Override to dispose kernels and free GPU memory.
    /// </summary>
    /// <example>
    /// <code>
    /// protected override async Task CleanupGpuResourcesAsync(CancellationToken cancellationToken)
    /// {
    ///     _myKernel?.Dispose();
    ///     await base.CleanupGpuResourcesAsync(cancellationToken);
    /// }
    /// </code>
    /// </example>
    protected virtual Task CleanupGpuResourcesAsync(CancellationToken cancellationToken)
    {
        // Default: No GPU resources to cleanup
        return Task.CompletedTask;
    }

    /// <summary>
    /// Executes a GPU kernel with automatic CPU fallback on failure.
    /// </summary>
    protected async Task<TOut> ExecuteKernelWithFallbackAsync<TIn, TOut>(
        IGpuKernel<TIn, TOut> kernel,
        TIn input,
        Func<TIn, Task<TOut>> cpuFallback,
        CancellationToken cancellationToken = default)
    {
        try
        {
            if (kernel.IsGpuAccelerated)
            {
                return await kernel.ExecuteAsync(input, cancellationToken);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(
                ex,
                "GPU kernel execution failed for {KernelId}, falling back to CPU",
                kernel.KernelId);
        }

        // CPU fallback
        return await cpuFallback(input);
    }

    /// <summary>
    /// Gets telemetry data for this grain's GPU usage.
    /// </summary>
    protected virtual GpuGrainTelemetry GetTelemetry()
    {
        return new GpuGrainTelemetry(
            GrainType: GetType().Name,
            GrainId: this.GetPrimaryKeyString(),
            GpuDeviceId: GpuDeviceId,
            IsGpuAccelerated: IsGpuAvailable,
            IsInitialized: _isInitialized);
    }
}

/// <summary>
/// Telemetry data for GPU grains
/// </summary>
public sealed record GpuGrainTelemetry(
    string GrainType,
    string GrainId,
    int GpuDeviceId,
    bool IsGpuAccelerated,
    bool IsInitialized);
