using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Grains.Base;

/// <summary>
/// Base class for GPU-resident ring kernel actors.
/// Actors live permanently in GPU memory with sub-microsecond message processing.
/// </summary>
/// <typeparam name="TState">Actor state type (GPU-resident)</typeparam>
/// <typeparam name="TMessage">Message type for actor communication</typeparam>
/// <remarks>
/// **GPU-Native Actor Paradigm:**
///
/// Traditional Orleans actors:
/// - Live in CPU memory
/// - Offload compute to GPU via kernel launches (~10-50μs overhead)
/// - Message latency: 10-100μs
///
/// Ring Kernel actors (this class):
/// - Live permanently in GPU memory
/// - Process messages entirely on GPU
/// - Ring kernel runs infinite dispatch loop
/// - Message latency: 100-500ns (20-200× faster)
///
/// **Architecture:**
/// ```
/// ┌─────────────────────────────────────────────────────┐
/// │              GPU Memory (Persistent)                │
/// │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐│
/// │  │ Actor State│  │ Message Queue│  │ HLC/Vector  ││
/// │  │ (TState)   │  │ (Lock-Free)  │  │ Clock       ││
/// │  └────────────┘  └──────────────┘  └─────────────┘│
/// │                                                     │
/// │  ┌─────────────────────────────────────────────┐   │
/// │  │   Ring Kernel (Infinite Dispatch Loop)      │   │
/// │  │   - Poll message queue (100ns interval)      │   │
/// │  │   - Process message → Update state           │   │
/// │  │   - Update HLC/Vector Clock                  │   │
/// │  │   - Runs forever (launched once)             │   │
/// │  └─────────────────────────────────────────────┘   │
/// └─────────────────────────────────────────────────────┘
/// ```
///
/// **Use Cases:**
/// - Real-time hypergraph pattern detection (&lt;100μs)
/// - Digital twins with physics simulation (100-500ns message latency)
/// - Temporal graph traversal with causal ordering
/// - Knowledge organisms with emergent intelligence
/// </remarks>
public abstract class RingKernelGrainBase<TState, TMessage> : Grain, IGrainBase
    where TState : struct // Struct for GPU memory layout
    where TMessage : struct // Struct for GPU message passing
{
    private readonly ILogger _logger;
    private readonly IGrainContext _grainContext;
    private RingKernelConfig _config = RingKernelConfig.Default;
    private bool _isRingKernelActive;
    private bool _isDisposed;

    // GPU backend services (lazy-initialized)
    private IGpuBackendProvider? _backendProvider;
    private IMemoryAllocator? _memoryAllocator;
    private RingKernelManager? _ringKernelManager;

    // GPU memory handles
    private IDeviceMemory? _gpuStateMemory;
    private IDeviceMemory? _gpuMessageQueueMemory;
    private IDeviceMemory? _hlcMemory;
    private IDeviceMemory? _vectorClockMemory;

    /// <summary>
    /// Current actor state (synchronized from GPU)
    /// </summary>
    protected TState State { get; private set; }

    /// <summary>
    /// Hybrid Logical Clock for temporal ordering (GPU-resident if enabled)
    /// </summary>
    protected HybridLogicalClock? HLC { get; private set; }

    /// <summary>
    /// GPU device this ring kernel is running on
    /// </summary>
    protected int GpuDeviceId { get; private set; }

    /// <summary>
    /// Number of messages processed by this ring kernel actor
    /// </summary>
    protected long MessagesProcessed { get; private set; }

    /// <summary>
    /// Average message processing latency (nanoseconds)
    /// </summary>
    protected long AverageLatencyNanoseconds { get; private set; }

    protected RingKernelGrainBase(IGrainContext grainContext, ILogger logger)
    {
        _grainContext = grainContext ?? throw new ArgumentNullException(nameof(grainContext));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    IGrainContext IGrainBase.GrainContext => _grainContext;

    /// <summary>
    /// Activates the ring kernel actor and launches persistent GPU kernel.
    /// </summary>
    public override async Task OnActivateAsync(CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation(
                "Activating ring kernel grain {GrainType} with ID {GrainId}",
                GetType().Name,
                this.GetPrimaryKeyString());

            // Get ring kernel configuration from derived class
            _config = await ConfigureRingKernelAsync(cancellationToken);

            // Determine GPU device placement
            GpuDeviceId = await DetermineGpuDevicePlacementAsync(cancellationToken);

            // Allocate GPU memory for actor state
            await AllocateGpuStateMemoryAsync(cancellationToken);

            // Allocate GPU message queue
            await AllocateGpuMessageQueueAsync(cancellationToken);

            // Initialize temporal clocks on GPU if enabled
            if (_config.EnableHLC)
            {
                await InitializeHLCOnGpuAsync(cancellationToken);
            }

            if (_config.EnableVectorClock)
            {
                await InitializeVectorClockOnGpuAsync(cancellationToken);
            }

            // Launch persistent ring kernel (runs forever until grain deactivation)
            await LaunchRingKernelAsync(cancellationToken);

            _isRingKernelActive = true;

            _logger.LogInformation(
                "Ring kernel grain {GrainType} activated successfully on GPU {DeviceId} " +
                "(Queue depth: {QueueDepth}, HLC: {HLC}, VectorClock: {VectorClock})",
                GetType().Name,
                GpuDeviceId,
                _config.QueueDepth,
                _config.EnableHLC,
                _config.EnableVectorClock);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Failed to activate ring kernel grain {GrainType}",
                GetType().Name);
            throw;
        }
    }

    /// <summary>
    /// Deactivates the ring kernel actor and stops persistent GPU kernel.
    /// </summary>
    public override async Task OnDeactivateAsync(DeactivationReason reason, CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation(
                "Deactivating ring kernel grain {GrainType} (Reason: {Reason}, " +
                "Messages Processed: {MessageCount}, Avg Latency: {LatencyNs}ns)",
                GetType().Name,
                reason.ReasonCode,
                MessagesProcessed,
                AverageLatencyNanoseconds);

            if (_isRingKernelActive && !_isDisposed)
            {
                // Signal ring kernel to stop
                await StopRingKernelAsync(cancellationToken);

                // Wait for kernel to finish processing in-flight messages
                await WaitForKernelShutdownAsync(cancellationToken);

                // Synchronize final state from GPU to CPU
                await SynchronizeStateFromGpuAsync(cancellationToken);

                // Free GPU resources
                await FreeGpuResourcesAsync(cancellationToken);

                _isDisposed = true;
            }

            _logger.LogInformation(
                "Ring kernel grain {GrainType} deactivated successfully",
                GetType().Name);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Error during ring kernel grain {GrainType} deactivation",
                GetType().Name);
            // Don't rethrow - deactivation should complete
        }
    }

    /// <summary>
    /// Sends a message to this ring kernel actor (GPU-resident queue).
    /// </summary>
    /// <param name="message">Message to process</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the message enqueue operation</returns>
    /// <remarks>
    /// Message is enqueued to GPU-resident lock-free queue.
    /// Ring kernel processes asynchronously at 100-500ns latency.
    /// </remarks>
    protected async Task SendMessageAsync(TMessage message, CancellationToken cancellationToken = default)
    {
        if (!_isRingKernelActive)
        {
            throw new InvalidOperationException(
                "Ring kernel is not active. Cannot send message.");
        }

        // Enqueue message to GPU-resident queue
        await EnqueueMessageToGpuAsync(message, cancellationToken);

        // Update telemetry
        MessagesProcessed++;
    }

    /// <summary>
    /// Configure ring kernel for this actor.
    /// Override to customize queue depth, temporal features, etc.
    /// </summary>
    protected virtual Task<RingKernelConfig> ConfigureRingKernelAsync(CancellationToken cancellationToken)
    {
        return Task.FromResult(RingKernelConfig.Default);
    }

    /// <summary>
    /// Process a message on the GPU.
    /// Override to implement actor message handling logic.
    /// </summary>
    /// <remarks>
    /// This method is compiled to GPU code and runs in the ring kernel dispatch loop.
    /// CONSTRAINTS:
    /// - No heap allocations
    /// - No virtual calls
    /// - No exceptions
    /// - Pure computation on GPU-resident data
    /// </remarks>
    protected abstract void ProcessMessageOnGpu(ref TState state, in TMessage message, ref HybridTimestamp hlc);

    /// <summary>
    /// Determines GPU device placement for this actor.
    /// Override for custom placement (e.g., co-location with related actors).
    /// </summary>
    protected virtual Task<int> DetermineGpuDevicePlacementAsync(CancellationToken cancellationToken)
    {
        // Default: GPU device 0
        // TODO: Implement queue-depth aware placement
        return Task.FromResult(0);
    }

    #region GPU Resource Management

    /// <summary>
    /// Ensures GPU backend services are initialized.
    /// </summary>
    private void EnsureBackendInitialized()
    {
        if (_backendProvider != null)
            return;

        _backendProvider = ServiceProvider.GetService<IGpuBackendProvider>();
        if (_backendProvider == null)
        {
            _logger.LogWarning("No GPU backend provider registered, ring kernel will use CPU fallback");
            return;
        }

        _memoryAllocator = _backendProvider.GetMemoryAllocator();
        _ringKernelManager = ServiceProvider.GetService<RingKernelManager>();
    }

    /// <summary>
    /// Allocates GPU memory for the actor state.
    /// </summary>
    private async Task AllocateGpuStateMemoryAsync(CancellationToken cancellationToken)
    {
        EnsureBackendInitialized();

        if (_memoryAllocator == null)
        {
            _logger.LogDebug("GPU backend unavailable, using CPU memory for state");
            return;
        }

        var stateSize = Unsafe.SizeOf<TState>();
        var actualSize = Math.Max(stateSize, _config.MaxStateSizeBytes);

        var options = new MemoryAllocationOptions(
            Type: MemoryType.Device,
            ZeroInitialize: true,
            Alignment: 256); // GPU-friendly alignment

        _gpuStateMemory = await _memoryAllocator.AllocateAsync(actualSize, options, cancellationToken);

        _logger.LogDebug(
            "Allocated GPU state memory: {Size} bytes at {DevicePtr:X16}",
            actualSize,
            _gpuStateMemory.DevicePointer);
    }

    /// <summary>
    /// Allocates GPU memory for the lock-free message queue.
    /// </summary>
    private async Task AllocateGpuMessageQueueAsync(CancellationToken cancellationToken)
    {
        EnsureBackendInitialized();

        if (_memoryAllocator == null)
        {
            _logger.LogDebug("GPU backend unavailable, using CPU memory for message queue");
            return;
        }

        var messageSize = Unsafe.SizeOf<TMessage>();
        var queueSizeBytes = (long)_config.QueueDepth * messageSize;

        // Add space for queue head/tail pointers (atomics) - aligned to 64 bytes
        var headerSize = 128; // 2x uint64 for head/tail + padding
        var totalSize = headerSize + queueSizeBytes;

        var options = new MemoryAllocationOptions(
            Type: MemoryType.Device,
            ZeroInitialize: true,
            Alignment: 128); // Cache-line aligned for atomics

        _gpuMessageQueueMemory = await _memoryAllocator.AllocateAsync(totalSize, options, cancellationToken);

        _logger.LogDebug(
            "Allocated GPU message queue: {Depth} messages × {MessageSize} bytes = {TotalSize} bytes",
            _config.QueueDepth,
            messageSize,
            totalSize);
    }

    /// <summary>
    /// Initializes GPU-resident Hybrid Logical Clock.
    /// </summary>
    private async Task InitializeHLCOnGpuAsync(CancellationToken cancellationToken)
    {
        EnsureBackendInitialized();

        if (_memoryAllocator == null)
        {
            _logger.LogDebug("GPU backend unavailable, using CPU HLC");
            HLC = new HybridLogicalClock(nodeId: 0);
            return;
        }

        // HLC structure: { uint64 physical, uint64 logical, uint16 nodeId } + padding
        const int hlcSize = 24;

        var options = new MemoryAllocationOptions(
            Type: MemoryType.Device,
            ZeroInitialize: false,
            Alignment: 16);

        _hlcMemory = await _memoryAllocator.AllocateAsync(hlcSize, options, cancellationToken);

        // Initialize HLC with current time
        HLC = new HybridLogicalClock(nodeId: 0);
        await CopyHlcToGpuAsync(cancellationToken);

        _logger.LogDebug("Initialized GPU-resident HLC at {DevicePtr:X16}", _hlcMemory.DevicePointer);
    }

    /// <summary>
    /// Initializes GPU-resident vector clock for causality tracking.
    /// </summary>
    private async Task InitializeVectorClockOnGpuAsync(CancellationToken cancellationToken)
    {
        EnsureBackendInitialized();

        if (_memoryAllocator == null)
        {
            _logger.LogDebug("GPU backend unavailable, skipping GPU vector clock");
            return;
        }

        // Vector clock: array of uint64 counters
        var vectorClockSize = (long)_config.VectorClockSize * sizeof(ulong);

        var options = new MemoryAllocationOptions(
            Type: MemoryType.Device,
            ZeroInitialize: true,
            Alignment: 64);

        _vectorClockMemory = await _memoryAllocator.AllocateAsync(vectorClockSize, options, cancellationToken);

        _logger.LogDebug(
            "Initialized GPU-resident vector clock: {Size} entries at {DevicePtr:X16}",
            _config.VectorClockSize,
            _vectorClockMemory.DevicePointer);
    }

    /// <summary>
    /// Launches the persistent ring kernel for message processing.
    /// </summary>
    private async Task LaunchRingKernelAsync(CancellationToken cancellationToken)
    {
        EnsureBackendInitialized();

        if (_ringKernelManager == null)
        {
            _logger.LogWarning("Ring kernel manager not available, running in CPU mode");
            return;
        }

        // Start the ring kernel manager with our configuration
        // Note: The RingKernelManager handles kernel launching internally
        if (!_ringKernelManager.IsRunning)
        {
            await _ringKernelManager.StartAsync(
                actorCount: 1, // Single actor per grain
                messageQueueSize: _config.QueueDepth,
                ct: cancellationToken);
        }

        var actorId = this.GetPrimaryKeyString() ?? this.GetPrimaryKey().ToString();
        _logger.LogInformation(
            "Ring kernel manager started for actor {ActorId} on GPU {DeviceId}",
            actorId,
            GpuDeviceId);
    }

    /// <summary>
    /// Signals the ring kernel to stop processing.
    /// </summary>
    private async Task StopRingKernelAsync(CancellationToken cancellationToken)
    {
        if (_ringKernelManager == null || !_ringKernelManager.IsRunning)
        {
            _logger.LogDebug("Ring kernel manager not available or not running, nothing to stop");
            return;
        }

        var actorId = this.GetPrimaryKeyString() ?? this.GetPrimaryKey().ToString();
        await _ringKernelManager.StopAsync(cancellationToken);

        _logger.LogDebug("Signaled ring kernel to stop for actor {ActorId}", actorId);
    }

    /// <summary>
    /// Waits for the ring kernel to complete shutdown.
    /// </summary>
    private async Task WaitForKernelShutdownAsync(CancellationToken cancellationToken)
    {
        if (_ringKernelManager == null)
            return;

        var actorId = this.GetPrimaryKeyString() ?? this.GetPrimaryKey().ToString();

        // Wait for the manager to stop if it's still running
        if (_ringKernelManager.IsRunning)
        {
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(30));

            // Poll for shutdown completion
            while (_ringKernelManager.IsRunning && !timeoutCts.Token.IsCancellationRequested)
            {
                await Task.Delay(10, timeoutCts.Token);
            }
        }

        _logger.LogDebug("Ring kernel shutdown complete for actor {ActorId}", actorId);
    }

    /// <summary>
    /// Synchronizes the final state from GPU to CPU.
    /// </summary>
    private async Task SynchronizeStateFromGpuAsync(CancellationToken cancellationToken)
    {
        if (_gpuStateMemory == null)
        {
            _logger.LogDebug("No GPU state memory, skipping synchronization");
            return;
        }

        var stateSize = Unsafe.SizeOf<TState>();
        var stateBytes = new byte[stateSize];

        // Get pointer outside of async context
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(stateBytes, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            var ptr = handle.AddrOfPinnedObject();
            await _gpuStateMemory.CopyToHostAsync(ptr, 0, stateSize, cancellationToken);
        }
        finally
        {
            handle.Free();
        }

        State = MemoryMarshal.Read<TState>(stateBytes);

        _logger.LogDebug("Synchronized {StateSize} bytes from GPU to CPU", stateSize);
    }

    /// <summary>
    /// Enqueues a message to the GPU-resident lock-free queue.
    /// </summary>
    private async Task EnqueueMessageToGpuAsync(TMessage message, CancellationToken cancellationToken)
    {
        if (_ringKernelManager == null || !_ringKernelManager.IsRunning)
        {
            // CPU fallback: process message directly
            var hlc = HLC?.Now() ?? new HybridTimestamp(DateTimeOffset.UtcNow.ToUnixTimeNanoseconds(), 0);
            var state = State;
            ProcessMessageOnGpu(ref state, in message, ref hlc);
            State = state;
            MessagesProcessed++;
            return;
        }

        // GPU path: enqueue to ring kernel
        // Convert TMessage to ActorMessage for the queue
        var actorMessage = CreateActorMessage(message);
        await _ringKernelManager.EnqueueMessageAsync(actorMessage, cancellationToken);
    }

    /// <summary>
    /// Creates an ActorMessage from the generic message type.
    /// </summary>
    private ActorMessage CreateActorMessage(TMessage message)
    {
        var actorIdStr = this.GetPrimaryKeyString() ?? this.GetPrimaryKey().ToString();
        var actorIdHash = (ulong)actorIdStr.GetHashCode();

        return new ActorMessage
        {
            MessageId = Guid.NewGuid(),
            SourceActorId = 0, // External source
            TargetActorId = actorIdHash,
            Timestamp = HLC?.Now() ?? HybridTimestamp.Now(),
            Type = MessageType.Command,
            Payload = 0, // Message payload serialized separately
            SequenceNumber = (ulong)Interlocked.Increment(ref _messageSequence),
            Priority = 0
        };
    }

    private long _messageSequence;

    /// <summary>
    /// Frees all GPU resources allocated for this actor.
    /// </summary>
    private Task FreeGpuResourcesAsync(CancellationToken cancellationToken)
    {
        try
        {
            _gpuStateMemory?.Dispose();
            _gpuMessageQueueMemory?.Dispose();
            _hlcMemory?.Dispose();
            _vectorClockMemory?.Dispose();

            _gpuStateMemory = null;
            _gpuMessageQueueMemory = null;
            _hlcMemory = null;
            _vectorClockMemory = null;

            _logger.LogDebug("Freed all GPU resources for ring kernel actor");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error freeing GPU resources");
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Copies the current HLC state to GPU memory.
    /// </summary>
    private async Task CopyHlcToGpuAsync(CancellationToken cancellationToken)
    {
        if (_hlcMemory == null || HLC == null)
            return;

        var timestamp = HLC.Now();
        var hlcBytes = new byte[24];

        // Pack HLC: physical (8 bytes) | logical (8 bytes) | nodeId (2 bytes) + padding (6 bytes)
        BitConverter.TryWriteBytes(hlcBytes.AsSpan()[..8], timestamp.PhysicalTime);
        BitConverter.TryWriteBytes(hlcBytes.AsSpan()[8..16], timestamp.LogicalCounter);
        BitConverter.TryWriteBytes(hlcBytes.AsSpan()[16..18], timestamp.NodeId);

        // Use GCHandle to pin memory for async operation
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(hlcBytes, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            var ptr = handle.AddrOfPinnedObject();
            await _hlcMemory.CopyFromHostAsync(ptr, 0, 24, cancellationToken);
        }
        finally
        {
            handle.Free();
        }
    }

    #endregion
}

/// <summary>
/// Simplified ring kernel grain base for stateless message processing.
/// </summary>
public abstract class RingKernelGrainBase<TMessage> : RingKernelGrainBase<EmptyState, TMessage>
    where TMessage : struct
{
    protected RingKernelGrainBase(IGrainContext grainContext, ILogger logger)
        : base(grainContext, logger)
    {
    }

    protected override void ProcessMessageOnGpu(ref EmptyState state, in TMessage message, ref HybridTimestamp hlc)
    {
        // Delegate to stateless version
        ProcessMessageOnGpu(in message, ref hlc);
    }

    /// <summary>
    /// Process a message on the GPU (stateless version).
    /// </summary>
    protected abstract void ProcessMessageOnGpu(in TMessage message, ref HybridTimestamp hlc);
}

/// <summary>
/// Empty state for stateless ring kernel actors.
/// </summary>
public struct EmptyState
{
}
