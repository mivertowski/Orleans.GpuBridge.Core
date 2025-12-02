// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.RingKernels;

/// <summary>
/// Marker interface for CPU fallback handlers.
/// </summary>
/// <remarks>
/// <para>
/// CPU fallback handlers provide equivalent logic to GPU ring kernel handlers
/// but execute on the CPU. This enables:
/// </para>
/// <list type="bullet">
/// <item><description>Development without GPU hardware</description></item>
/// <item><description>Testing in CI/CD environments</description></item>
/// <item><description>Graceful degradation when GPU is unavailable</description></item>
/// </list>
/// </remarks>
public interface ICpuFallbackHandler
{
    /// <summary>
    /// Gets the kernel ID this handler is associated with.
    /// </summary>
    string KernelId { get; }

    /// <summary>
    /// Gets the handler ID within the kernel.
    /// </summary>
    int HandlerId { get; }

    /// <summary>
    /// Gets a human-readable description of this handler.
    /// </summary>
    string Description { get; }
}

/// <summary>
/// Generic interface for CPU fallback handlers that process request/response messages.
/// </summary>
/// <typeparam name="TRequest">Request message type (must be unmanaged/blittable).</typeparam>
/// <typeparam name="TResponse">Response message type (must be unmanaged/blittable).</typeparam>
/// <typeparam name="TState">Actor state type (must be unmanaged/blittable).</typeparam>
/// <remarks>
/// <para>
/// Implementations should mirror the logic of the GPU ring kernel handler.
/// The handler receives the request and current state, and returns the response
/// and potentially updated state.
/// </para>
/// <para>
/// <b>Thread Safety:</b> Handlers may be called concurrently from multiple threads.
/// Implementations must be thread-safe or stateless.
/// </para>
/// <para>
/// <b>Performance:</b> CPU fallback is typically 10-100x slower than GPU execution.
/// Expected latency: 1-10μs per handler invocation.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// public class VectorAddCpuHandler : ICpuFallbackHandler&lt;VectorAddRequest, VectorAddResponse, VectorAddState&gt;
/// {
///     public string KernelId =&gt; "vectoradd_processor";
///     public int HandlerId =&gt; 0;
///     public string Description =&gt; "CPU fallback for vector addition";
///
///     public (VectorAddResponse, VectorAddState) Execute(VectorAddRequest request, VectorAddState state)
///     {
///         var response = new VectorAddResponse
///         {
///             R0 = request.A0 + request.B0,
///             R1 = request.A1 + request.B1,
///             // ...
///         };
///         return (response, state);
///     }
/// }
/// </code>
/// </example>
public interface ICpuFallbackHandler<TRequest, TResponse, TState> : ICpuFallbackHandler
    where TRequest : unmanaged
    where TResponse : unmanaged
    where TState : unmanaged
{
    /// <summary>
    /// Executes the handler logic on the CPU.
    /// </summary>
    /// <param name="request">The request message.</param>
    /// <param name="currentState">The current actor state.</param>
    /// <returns>A tuple of (response message, updated state).</returns>
    (TResponse Response, TState NewState) Execute(TRequest request, TState currentState);
}

/// <summary>
/// Generic interface for fire-and-forget CPU fallback handlers.
/// </summary>
/// <typeparam name="TRequest">Request message type (must be unmanaged/blittable).</typeparam>
/// <typeparam name="TState">Actor state type (must be unmanaged/blittable).</typeparam>
/// <remarks>
/// <para>
/// Fire-and-forget handlers don't return a response. They are used for
/// operations like state updates, event notifications, or metrics collection.
/// </para>
/// </remarks>
public interface ICpuFallbackFireAndForgetHandler<TRequest, TState> : ICpuFallbackHandler
    where TRequest : unmanaged
    where TState : unmanaged
{
    /// <summary>
    /// Executes the fire-and-forget handler logic on the CPU.
    /// </summary>
    /// <param name="request">The request message.</param>
    /// <param name="currentState">The current actor state.</param>
    /// <returns>The updated state.</returns>
    TState Execute(TRequest request, TState currentState);
}

/// <summary>
/// Stateless CPU fallback handler interface for request/response processing.
/// </summary>
/// <typeparam name="TRequest">Request message type.</typeparam>
/// <typeparam name="TResponse">Response message type.</typeparam>
/// <remarks>
/// Use this interface for handlers that don't need to access or modify state.
/// </remarks>
public interface IStatelessCpuFallbackHandler<TRequest, TResponse> : ICpuFallbackHandler
    where TRequest : unmanaged
    where TResponse : unmanaged
{
    /// <summary>
    /// Executes the stateless handler logic.
    /// </summary>
    /// <param name="request">The request message.</param>
    /// <returns>The response message.</returns>
    TResponse Execute(TRequest request);
}

/// <summary>
/// Attribute to mark a CPU fallback handler implementation.
/// </summary>
/// <remarks>
/// This attribute enables automatic discovery of CPU fallback handlers during startup.
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class CpuFallbackHandlerAttribute : Attribute
{
    /// <summary>
    /// Gets the kernel ID this handler is associated with.
    /// </summary>
    public string KernelId { get; }

    /// <summary>
    /// Gets the handler ID within the kernel.
    /// </summary>
    public int HandlerId { get; }

    /// <summary>
    /// Creates a new CPU fallback handler attribute.
    /// </summary>
    /// <param name="kernelId">The kernel ID.</param>
    /// <param name="handlerId">The handler ID within the kernel (default: 0).</param>
    public CpuFallbackHandlerAttribute(string kernelId, int handlerId = 0)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelId);
        KernelId = kernelId;
        HandlerId = handlerId;
    }
}
