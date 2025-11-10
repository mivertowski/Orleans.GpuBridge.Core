using System;
using DotCompute.Memory;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Adapter for DotCompute's IMemoryOrderingProvider - enables causal memory ordering.
/// Provides acquire-release semantics and memory fence injection for distributed actor correctness.
/// </summary>
public sealed class DotComputeMemoryOrderingProvider : IDisposable
{
    private readonly IMemoryOrderingProvider _dotComputeMemoryOrdering;
    private readonly ILogger<DotComputeMemoryOrderingProvider> _logger;
    private bool _causalOrderingEnabled;
    private MemoryConsistencyModel _currentModel;
    private bool _disposed;

    public DotComputeMemoryOrderingProvider(
        IMemoryOrderingProvider dotComputeMemoryOrdering,
        ILogger<DotComputeMemoryOrderingProvider> logger)
    {
        _dotComputeMemoryOrdering = dotComputeMemoryOrdering ?? throw new ArgumentNullException(nameof(dotComputeMemoryOrdering));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _currentModel = _dotComputeMemoryOrdering.ConsistencyModel;

        _logger.LogInformation(
            "DotComputeMemoryOrderingProvider initialized - Acquire-release support: {AcquireReleaseSupport}, Model: {Model}",
            _dotComputeMemoryOrdering.IsAcquireReleaseSupported,
            _currentModel);
    }

    /// <summary>
    /// Checks if device supports acquire-release semantics.
    /// Required for causal ordering in distributed actors.
    /// </summary>
    public bool IsAcquireReleaseSupported => _dotComputeMemoryOrdering.IsAcquireReleaseSupported;

    /// <summary>
    /// Gets current memory consistency model.
    /// </summary>
    public MemoryConsistencyModel ConsistencyModel => _currentModel;

    /// <summary>
    /// Checks if causal ordering is currently enabled.
    /// </summary>
    public bool IsCausalOrderingEnabled => _causalOrderingEnabled;

    /// <summary>
    /// Enables causal memory ordering for all memory operations.
    /// When enabled:
    /// - Writes use release semantics (make prior writes visible)
    /// - Reads use acquire semantics (make subsequent operations wait)
    /// - Memory fences enforce ordering
    /// Performance impact: ~15% overhead vs relaxed consistency
    /// </summary>
    public void EnableCausalOrdering(bool enable = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!IsAcquireReleaseSupported && enable)
        {
            throw new InvalidOperationException(
                "Device does not support acquire-release semantics required for causal ordering");
        }

        try
        {
            _dotComputeMemoryOrdering.EnableCausalOrdering(enable);
            _causalOrderingEnabled = enable;

            _logger.LogInformation(
                "Causal ordering {Status} - Performance impact: ~{Impact}%",
                enable ? "enabled" : "disabled",
                enable ? 15 : 0);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to configure causal ordering");
            throw;
        }
    }

    /// <summary>
    /// Sets memory consistency model.
    /// Relaxed: Default GPU model (fastest, no ordering guarantees)
    /// ReleaseAcquire: Causal ordering (15% overhead)
    /// Sequential: Total order (40% overhead, rarely needed)
    /// </summary>
    public void SetConsistencyModel(MemoryConsistencyModel model)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (model == MemoryConsistencyModel.ReleaseAcquire && !IsAcquireReleaseSupported)
        {
            throw new InvalidOperationException(
                "Device does not support release-acquire consistency model");
        }

        try
        {
            _dotComputeMemoryOrdering.SetConsistencyModel(model);
            _currentModel = model;

            var overhead = model switch
            {
                MemoryConsistencyModel.Relaxed => 0,
                MemoryConsistencyModel.ReleaseAcquire => 15,
                MemoryConsistencyModel.Sequential => 40,
                _ => 0
            };

            _logger.LogInformation(
                "Memory consistency model set to {Model} - Performance overhead: ~{Overhead}%",
                model, overhead);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to set consistency model to {Model}", model);
            throw;
        }
    }

    /// <summary>
    /// Inserts memory fence at specified location in kernel.
    /// CUDA: __threadfence_block() (10ns), __threadfence() (100ns), __threadfence_system() (200ns)
    /// OpenCL: mem_fence(), atomic_work_item_fence()
    /// </summary>
    public void InsertFence(FenceType type, FenceLocation? location = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        try
        {
            _dotComputeMemoryOrdering.InsertFence(type, location);

            var overhead = type switch
            {
                FenceType.ThreadBlock => 10,
                FenceType.Device => 100,
                FenceType.System => 200,
                _ => 0
            };

            _logger.LogDebug(
                "Inserted {FenceType} fence at {Location} - Overhead: ~{Overhead}ns",
                type,
                location != null ? "custom location" : "default location",
                overhead);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to insert {FenceType} fence", type);
            throw;
        }
    }

    /// <summary>
    /// Inserts fence before all memory reads (acquire semantics).
    /// Use for: Reading message data after observing timestamp.
    /// </summary>
    public void InsertAcquireFence(FenceType type = FenceType.Device)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var location = new FenceLocation { BeforeReads = true };
        InsertFence(type, location);

        _logger.LogDebug("Inserted acquire fence (before reads) - Type: {FenceType}", type);
    }

    /// <summary>
    /// Inserts fence after all memory writes (release semantics).
    /// Use for: Writing message data before publishing timestamp.
    /// </summary>
    public void InsertReleaseFence(FenceType type = FenceType.Device)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var location = new FenceLocation { AfterWrites = true };
        InsertFence(type, location);

        _logger.LogDebug("Inserted release fence (after writes) - Type: {FenceType}", type);
    }

    /// <summary>
    /// Inserts fences at kernel entry and exit for complete isolation.
    /// Use for: Ring kernels that need strict ordering with host.
    /// </summary>
    public void InsertKernelBoundaryFences(FenceType type = FenceType.System)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Entry fence
        InsertFence(type, new FenceLocation { AtEntry = true });

        // Exit fence
        InsertFence(type, new FenceLocation { AtExit = true });

        _logger.LogDebug(
            "Inserted kernel boundary fences (entry + exit) - Type: {FenceType}",
            type);
    }

    /// <summary>
    /// Configures memory ordering for GPU-native actor message passing.
    /// Ensures messages are visible across actors in causal order.
    /// </summary>
    public void ConfigureActorMessageOrdering()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        try
        {
            _logger.LogInformation("Configuring memory ordering for actor message passing...");

            // Use release-acquire model for causal ordering
            SetConsistencyModel(MemoryConsistencyModel.ReleaseAcquire);

            // Enable causal ordering
            EnableCausalOrdering(true);

            // Insert release fence after message writes
            InsertReleaseFence(FenceType.System);

            // Insert acquire fence before message reads
            InsertAcquireFence(FenceType.System);

            _logger.LogInformation(
                "Actor message ordering configured - " +
                "Model: ReleaseAcquire, Fences: System-level, Overhead: ~15%");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to configure actor message ordering");
            throw;
        }
    }

    /// <summary>
    /// Resets memory ordering to default (relaxed) for maximum performance.
    /// Use when causal ordering is not required.
    /// </summary>
    public void ResetToDefault()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        try
        {
            SetConsistencyModel(MemoryConsistencyModel.Relaxed);
            EnableCausalOrdering(false);

            _logger.LogInformation("Memory ordering reset to default (relaxed) - Maximum performance");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to reset memory ordering to default");
            throw;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            // Reset to default on cleanup
            if (_dotComputeMemoryOrdering != null)
            {
                _dotComputeMemoryOrdering.SetConsistencyModel(MemoryConsistencyModel.Relaxed);
                _dotComputeMemoryOrdering.EnableCausalOrdering(false);
            }

            _disposed = true;
            _logger.LogDebug("DotComputeMemoryOrderingProvider disposed");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error during DotComputeMemoryOrderingProvider disposal");
        }
    }
}
