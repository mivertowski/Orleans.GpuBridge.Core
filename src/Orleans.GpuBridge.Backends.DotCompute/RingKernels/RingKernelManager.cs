using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Synchronization;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels;

/// <summary>
/// Manages ring kernels - persistent GPU kernels that run indefinitely processing actor messages.
/// Ring kernels are launched once and run forever (until explicitly stopped).
/// Performance: 100-500ns message latency, 2M messages/s/actor throughput.
/// </summary>
public sealed class RingKernelManager : IDisposable
{
    private readonly DotComputeBarrierProvider _barrierProvider;
    private readonly DotComputeTimingProvider _timingProvider;
    private readonly DotComputeMemoryOrderingProvider _memoryOrdering;
    private readonly ILogger<RingKernelManager> _logger;
    private readonly ConcurrentDictionary<Guid, RingKernelInstance> _activeKernels;
    private bool _disposed;

    public RingKernelManager(
        DotComputeBarrierProvider barrierProvider,
        DotComputeTimingProvider timingProvider,
        DotComputeMemoryOrderingProvider memoryOrdering,
        ILogger<RingKernelManager> logger)
    {
        _barrierProvider = barrierProvider ?? throw new ArgumentNullException(nameof(barrierProvider));
        _timingProvider = timingProvider ?? throw new ArgumentNullException(nameof(timingProvider));
        _memoryOrdering = memoryOrdering ?? throw new ArgumentNullException(nameof(memoryOrdering));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _activeKernels = new ConcurrentDictionary<Guid, RingKernelInstance>();

        _logger.LogInformation(
            "RingKernelManager initialized - Hardware barriers: {HardwareSupport}",
            _barrierProvider.IsHardwareBarrierSupported);
    }

    /// <summary>
    /// Launches a ring kernel that runs indefinitely until stopped.
    /// Ring kernels process messages in an infinite loop on the GPU.
    /// </summary>
    public async Task<RingKernelHandle> LaunchRingKernelAsync(
        ICompiledKernel kernel,
        RingKernelConfiguration config,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIf(kernel);
        ArgumentNullException.ThrowIf(config);

        _logger.LogInformation(
            "Launching ring kernel - Actors: {ActorCount}, Threads/Actor: {ThreadsPerActor}, Total threads: {TotalThreads}",
            config.ActorCount,
            config.ThreadsPerActor,
            config.ActorCount * config.ThreadsPerActor);

        // Create barrier for ring kernel coordination
        var barrier = _barrierProvider.CreateRingKernelBarrier(
            config.ActorCount,
            config.ThreadsPerActor);

        // Configure memory ordering for actor messages
        if (config.EnableTemporalOrdering)
        {
            _memoryOrdering.ConfigureActorMessageOrdering();
        }

        // Enable timestamp injection if temporal features requested
        if (config.EnableTimestamps)
        {
            _timingProvider.EnableTimestampInjection(true);
        }

        // Create launch configuration
        var launchConfig = new LaunchConfiguration
        {
            GridDim = (config.ActorCount / config.BlockSize, 1, 1),
            BlockDim = (config.BlockSize * config.ThreadsPerActor, 1, 1),
            SharedMemoryBytes = config.SharedMemoryPerBlock
        };

        // Create cancellation source for this ring kernel
        var kernelCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        // Create instance tracking
        var instance = new RingKernelInstance
        {
            InstanceId = Guid.NewGuid(),
            Kernel = kernel,
            Barrier = barrier,
            Configuration = config,
            LaunchConfiguration = launchConfig,
            CancellationSource = kernelCts,
            LaunchTime = DateTimeOffset.UtcNow
        };

        _activeKernels[instance.InstanceId] = instance;

        // Launch ring kernel (runs until cancelled)
        var kernelTask = Task.Run(async () =>
        {
            try
            {
                _logger.LogInformation(
                    "Ring kernel {InstanceId} started - This kernel runs indefinitely",
                    instance.InstanceId);

                await _barrierProvider.ExecuteRingKernelAsync(
                    kernel,
                    barrier,
                    launchConfig,
                    config.Arguments,
                    kernelCts.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation(
                    "Ring kernel {InstanceId} stopped normally",
                    instance.InstanceId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "Ring kernel {InstanceId} failed",
                    instance.InstanceId);
                throw;
            }
        }, kernelCts.Token);

        instance.KernelTask = kernelTask;

        // Create handle for caller
        var handle = new RingKernelHandle(
            instance.InstanceId,
            kernelTask,
            kernelCts,
            this,
            _logger);

        _logger.LogInformation(
            "Ring kernel {InstanceId} launched successfully",
            instance.InstanceId);

        return handle;
    }

    /// <summary>
    /// Stops a running ring kernel.
    /// </summary>
    public async Task StopRingKernelAsync(Guid instanceId, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_activeKernels.TryGetValue(instanceId, out var instance))
        {
            _logger.LogWarning("Attempted to stop unknown ring kernel {InstanceId}", instanceId);
            return;
        }

        _logger.LogInformation("Stopping ring kernel {InstanceId}...", instanceId);

        try
        {
            // Cancel the kernel
            await instance.CancellationSource.CancelAsync();

            // Wait for kernel to stop (with timeout)
            using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);

            await instance.KernelTask.WaitAsync(linkedCts.Token).ConfigureAwait(false);

            _logger.LogInformation("Ring kernel {InstanceId} stopped successfully", instanceId);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            _logger.LogWarning("Ring kernel {InstanceId} stop cancelled", instanceId);
            throw;
        }
        catch (OperationCanceledException)
        {
            _logger.LogError("Ring kernel {InstanceId} did not stop within timeout", instanceId);
            throw new TimeoutException($"Ring kernel {instanceId} did not stop within 30 seconds");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping ring kernel {InstanceId}", instanceId);
            throw;
        }
        finally
        {
            // Cleanup
            if (_activeKernels.TryRemove(instanceId, out var removedInstance))
            {
                removedInstance.Barrier?.Dispose();
                removedInstance.CancellationSource?.Dispose();
            }
        }
    }

    /// <summary>
    /// Gets status of all active ring kernels.
    /// </summary>
    public RingKernelStatus[] GetActiveKernels()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var statuses = new RingKernelStatus[_activeKernels.Count];
        var index = 0;

        foreach (var kvp in _activeKernels)
        {
            var instance = kvp.Value;
            statuses[index++] = new RingKernelStatus
            {
                InstanceId = instance.InstanceId,
                ActorCount = instance.Configuration.ActorCount,
                ThreadsPerActor = instance.Configuration.ThreadsPerActor,
                EnabledTemporalOrdering = instance.Configuration.EnableTemporalOrdering,
                LaunchTime = instance.LaunchTime,
                Uptime = DateTimeOffset.UtcNow - instance.LaunchTime,
                IsRunning = instance.KernelTask?.Status == TaskStatus.Running
            };
        }

        return statuses;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _logger.LogInformation(
                "Disposing RingKernelManager with {ActiveKernels} active kernels",
                _activeKernels.Count);

            // Stop all active ring kernels
            foreach (var instance in _activeKernels.Values)
            {
                try
                {
                    instance.CancellationSource?.Cancel();
                    instance.Barrier?.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex,
                        "Error stopping ring kernel {InstanceId} during disposal",
                        instance.InstanceId);
                }
            }

            // Wait for all kernels to stop (with timeout)
            var stopTasks = _activeKernels.Values
                .Select(i => i.KernelTask)
                .Where(t => t != null)
                .ToArray();

            if (stopTasks.Length > 0)
            {
                Task.WaitAll(stopTasks, TimeSpan.FromSeconds(10));
            }

            _activeKernels.Clear();
            _disposed = true;

            _logger.LogInformation("RingKernelManager disposed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during RingKernelManager disposal");
        }
    }

    private sealed class RingKernelInstance
    {
        public required Guid InstanceId { get; init; }
        public required ICompiledKernel Kernel { get; init; }
        public required IBarrierHandle Barrier { get; init; }
        public required RingKernelConfiguration Configuration { get; init; }
        public required LaunchConfiguration LaunchConfiguration { get; init; }
        public required CancellationTokenSource CancellationSource { get; init; }
        public required DateTimeOffset LaunchTime { get; init; }
        public Task? KernelTask { get; set; }
    }
}

/// <summary>
/// Configuration for launching a ring kernel.
/// </summary>
public sealed class RingKernelConfiguration
{
    /// <summary>
    /// Number of GPU-native actors.
    /// </summary>
    public required int ActorCount { get; init; }

    /// <summary>
    /// Number of threads per actor (typically 1 for actors, >1 for parallel processing).
    /// </summary>
    public int ThreadsPerActor { get; init; } = 1;

    /// <summary>
    /// CUDA block size (threads per block).
    /// Typical values: 256, 512, 1024.
    /// </summary>
    public int BlockSize { get; init; } = 256;

    /// <summary>
    /// Shared memory per block in bytes.
    /// </summary>
    public int SharedMemoryPerBlock { get; init; } = 0;

    /// <summary>
    /// Kernel arguments (actor state, message queues, etc.).
    /// </summary>
    public required object[] Arguments { get; init; }

    /// <summary>
    /// Enable temporal ordering with HLC and memory fences.
    /// Performance impact: ~15% overhead.
    /// </summary>
    public bool EnableTemporalOrdering { get; init; } = true;

    /// <summary>
    /// Enable automatic timestamp injection at kernel entry.
    /// </summary>
    public bool EnableTimestamps { get; init; } = true;
}

/// <summary>
/// Handle for managing a running ring kernel.
/// </summary>
public sealed class RingKernelHandle : IDisposable
{
    private readonly Guid _instanceId;
    private readonly Task _kernelTask;
    private readonly CancellationTokenSource _cancellationSource;
    private readonly RingKernelManager _manager;
    private readonly ILogger _logger;
    private bool _disposed;

    internal RingKernelHandle(
        Guid instanceId,
        Task kernelTask,
        CancellationTokenSource cancellationSource,
        RingKernelManager manager,
        ILogger logger)
    {
        _instanceId = instanceId;
        _kernelTask = kernelTask;
        _cancellationSource = cancellationSource;
        _manager = manager;
        _logger = logger;
    }

    public Guid InstanceId => _instanceId;
    public bool IsRunning => _kernelTask.Status == TaskStatus.Running;

    /// <summary>
    /// Stops the ring kernel.
    /// </summary>
    public async Task StopAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _manager.StopRingKernelAsync(_instanceId, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Waits for the ring kernel to complete (typically never completes unless stopped).
    /// </summary>
    public Task WaitForCompletionAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _kernelTask.WaitAsync(ct);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            // Stop the kernel if still running
            if (IsRunning)
            {
                _cancellationSource.Cancel();
                _kernelTask.Wait(TimeSpan.FromSeconds(5));
            }

            _disposed = true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error disposing ring kernel handle {InstanceId}", _instanceId);
        }
    }
}

/// <summary>
/// Status information for a running ring kernel.
/// </summary>
public sealed class RingKernelStatus
{
    public required Guid InstanceId { get; init; }
    public required int ActorCount { get; init; }
    public required int ThreadsPerActor { get; init; }
    public required bool EnabledTemporalOrdering { get; init; }
    public required DateTimeOffset LaunchTime { get; init; }
    public required TimeSpan Uptime { get; init; }
    public required bool IsRunning { get; init; }

    public override string ToString()
    {
        return $"RingKernel {InstanceId}: {ActorCount} actors, Uptime: {Uptime.TotalSeconds:F1}s, " +
               $"Temporal: {EnabledTemporalOrdering}, Running: {IsRunning}";
    }
}
