// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.RingKernels;

/// <summary>
/// Bridge interface for executing ring kernel handlers on GPU or CPU.
/// </summary>
/// <remarks>
/// <para>
/// This interface abstracts the GPU execution layer from generated actors.
/// Implementations handle:
/// </para>
/// <list type="bullet">
/// <item><description>Ring kernel discovery and compilation</description></item>
/// <item><description>GPU memory allocation for actor state</description></item>
/// <item><description>Message serialization to/from GPU memory</description></item>
/// <item><description>Kernel dispatch and result collection</description></item>
/// </list>
/// <para>
/// <strong>Implementations:</strong>
/// </para>
/// <list type="bullet">
/// <item><description><c>DotComputeRingKernelBridge</c> - Real GPU execution via DotCompute</description></item>
/// <item><description><c>CpuFallbackBridge</c> - CPU-only fallback for testing</description></item>
/// </list>
/// </remarks>
public interface IRingKernelBridge
{
    /// <summary>
    /// Checks if GPU execution is available.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>True if at least one GPU device is available</returns>
    Task<bool> IsGpuAvailableAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the GPU device ID to use for actor placement.
    /// </summary>
    /// <param name="actorKey">The actor's grain key for placement decisions</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>GPU device ID (0-based) or -1 if no GPU available</returns>
    Task<int> GetDevicePlacementAsync(string actorKey, CancellationToken cancellationToken = default);

    /// <summary>
    /// Allocates GPU memory for actor state.
    /// </summary>
    /// <typeparam name="TState">Actor state type (blittable struct)</typeparam>
    /// <param name="actorId">Unique actor identifier</param>
    /// <param name="deviceId">Target GPU device ID</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Handle to GPU-resident state memory</returns>
    Task<GpuStateHandle<TState>> AllocateStateAsync<TState>(
        string actorId,
        int deviceId,
        CancellationToken cancellationToken = default)
        where TState : unmanaged;

    /// <summary>
    /// Releases GPU memory for actor state.
    /// </summary>
    /// <typeparam name="TState">Actor state type</typeparam>
    /// <param name="handle">State handle to release</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task ReleaseStateAsync<TState>(
        GpuStateHandle<TState> handle,
        CancellationToken cancellationToken = default)
        where TState : unmanaged;

    /// <summary>
    /// Executes a ring kernel handler with request/response pattern.
    /// </summary>
    /// <typeparam name="TRequest">Request message type (blittable struct)</typeparam>
    /// <typeparam name="TResponse">Response message type (blittable struct)</typeparam>
    /// <typeparam name="TState">Actor state type (blittable struct)</typeparam>
    /// <param name="kernelId">Ring kernel identifier</param>
    /// <param name="handlerId">Handler ID within the kernel</param>
    /// <param name="request">Request payload</param>
    /// <param name="stateHandle">Handle to GPU-resident state</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Response and updated state from GPU execution</returns>
    Task<RingKernelResult<TResponse, TState>> ExecuteHandlerAsync<TRequest, TResponse, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        GpuStateHandle<TState> stateHandle,
        CancellationToken cancellationToken = default)
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged;

    /// <summary>
    /// Executes a fire-and-forget ring kernel handler.
    /// </summary>
    /// <typeparam name="TRequest">Request message type (blittable struct)</typeparam>
    /// <typeparam name="TState">Actor state type (blittable struct)</typeparam>
    /// <param name="kernelId">Ring kernel identifier</param>
    /// <param name="handlerId">Handler ID within the kernel</param>
    /// <param name="request">Request payload</param>
    /// <param name="stateHandle">Handle to GPU-resident state</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Updated state from GPU execution</returns>
    Task<TState> ExecuteFireAndForgetAsync<TRequest, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        GpuStateHandle<TState> stateHandle,
        CancellationToken cancellationToken = default)
        where TRequest : unmanaged
        where TState : unmanaged;

    /// <summary>
    /// Reads current state from GPU memory.
    /// </summary>
    /// <typeparam name="TState">Actor state type</typeparam>
    /// <param name="handle">State handle</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Current state value</returns>
    Task<TState> ReadStateAsync<TState>(
        GpuStateHandle<TState> handle,
        CancellationToken cancellationToken = default)
        where TState : unmanaged;

    /// <summary>
    /// Writes state to GPU memory.
    /// </summary>
    /// <typeparam name="TState">Actor state type</typeparam>
    /// <param name="handle">State handle</param>
    /// <param name="state">State value to write</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task WriteStateAsync<TState>(
        GpuStateHandle<TState> handle,
        TState state,
        CancellationToken cancellationToken = default)
        where TState : unmanaged;

    /// <summary>
    /// Gets telemetry data for ring kernel operations.
    /// </summary>
    /// <returns>Current telemetry snapshot</returns>
    RingKernelTelemetry GetTelemetry();
}

/// <summary>
/// Handle to GPU-resident actor state.
/// </summary>
/// <typeparam name="TState">Actor state type</typeparam>
public sealed class GpuStateHandle<TState> : IDisposable
    where TState : unmanaged
{
    /// <summary>
    /// Gets the actor identifier this state belongs to.
    /// </summary>
    public string ActorId { get; }

    /// <summary>
    /// Gets the GPU device ID where state is allocated.
    /// </summary>
    public int DeviceId { get; }

    /// <summary>
    /// Gets the size of the state in bytes.
    /// </summary>
    public int SizeBytes { get; }

    /// <summary>
    /// Gets the GPU memory pointer (IntPtr for native interop).
    /// </summary>
    public IntPtr GpuPointer { get; internal set; }

    /// <summary>
    /// Gets whether this handle is valid (not disposed).
    /// </summary>
    public bool IsValid => !_disposed && GpuPointer != IntPtr.Zero;

    /// <summary>
    /// Gets the shadow copy of state for CPU fallback.
    /// </summary>
    internal TState ShadowState { get; set; }

    private bool _disposed;

    /// <summary>
    /// Creates a new GPU state handle.
    /// </summary>
    public GpuStateHandle(string actorId, int deviceId, int sizeBytes)
    {
        ActorId = actorId ?? throw new ArgumentNullException(nameof(actorId));
        DeviceId = deviceId;
        SizeBytes = sizeBytes;
        ShadowState = default;
    }

    /// <summary>
    /// Disposes the GPU state handle.
    /// </summary>
    /// <remarks>
    /// Note: Actual GPU memory deallocation is handled by the bridge.
    /// This just marks the handle as disposed.
    /// </remarks>
    public void Dispose()
    {
        _disposed = true;
        GpuPointer = IntPtr.Zero;
    }
}

/// <summary>
/// Result of ring kernel handler execution.
/// </summary>
/// <typeparam name="TResponse">Response message type</typeparam>
/// <typeparam name="TState">Actor state type</typeparam>
public readonly struct RingKernelResult<TResponse, TState>
    where TResponse : unmanaged
    where TState : unmanaged
{
    /// <summary>
    /// Gets the response from the handler.
    /// </summary>
    public TResponse Response { get; }

    /// <summary>
    /// Gets the updated state after handler execution.
    /// </summary>
    public TState NewState { get; }

    /// <summary>
    /// Gets whether the execution was successful.
    /// </summary>
    public bool Success { get; }

    /// <summary>
    /// Gets the error code if execution failed (0 = success).
    /// </summary>
    public int ErrorCode { get; }

    /// <summary>
    /// Gets the execution latency in nanoseconds.
    /// </summary>
    public long LatencyNs { get; }

    /// <summary>
    /// Gets whether execution used GPU (vs CPU fallback).
    /// </summary>
    public bool WasGpuExecution { get; }

    /// <summary>
    /// Creates a successful result.
    /// </summary>
    public RingKernelResult(TResponse response, TState newState, long latencyNs, bool wasGpuExecution)
    {
        Response = response;
        NewState = newState;
        Success = true;
        ErrorCode = 0;
        LatencyNs = latencyNs;
        WasGpuExecution = wasGpuExecution;
    }

    /// <summary>
    /// Creates a failed result.
    /// </summary>
    public RingKernelResult(int errorCode, TState currentState)
    {
        Response = default;
        NewState = currentState;
        Success = false;
        ErrorCode = errorCode;
        LatencyNs = 0;
        WasGpuExecution = false;
    }
}

/// <summary>
/// Telemetry data for ring kernel operations.
/// </summary>
public sealed record RingKernelTelemetry
{
    /// <summary>
    /// Total ring kernel executions.
    /// </summary>
    public long TotalExecutions { get; init; }

    /// <summary>
    /// Executions on GPU.
    /// </summary>
    public long GpuExecutions { get; init; }

    /// <summary>
    /// Executions on CPU fallback.
    /// </summary>
    public long CpuFallbackExecutions { get; init; }

    /// <summary>
    /// Total bytes transferred to GPU.
    /// </summary>
    public long BytesToGpu { get; init; }

    /// <summary>
    /// Total bytes transferred from GPU.
    /// </summary>
    public long BytesFromGpu { get; init; }

    /// <summary>
    /// Average latency in nanoseconds.
    /// </summary>
    public double AverageLatencyNs { get; init; }

    /// <summary>
    /// Number of active state handles.
    /// </summary>
    public int ActiveStateHandles { get; init; }

    /// <summary>
    /// Total GPU memory allocated for states (bytes).
    /// </summary>
    public long AllocatedGpuMemoryBytes { get; init; }
}
