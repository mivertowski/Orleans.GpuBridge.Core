// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Generation;

/// <summary>
/// Marks a method in a [GpuNativeActor] interface as a GPU-accelerated handler.
/// The source generator will generate the corresponding ring kernel handler code.
/// </summary>
/// <remarks>
/// <para>
/// Methods marked with this attribute will be automatically transformed into:
/// <list type="bullet">
/// <item><description>Request/Response message structs (blittable)</description></item>
/// <item><description>[RingKernel] handler method in a static class</description></item>
/// <item><description>Grain implementation that marshals calls to the kernel</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Requirements:</strong>
/// <list type="bullet">
/// <item><description>Method must return Task or Task&lt;T&gt;</description></item>
/// <item><description>All parameters must be blittable (value types without references)</description></item>
/// <item><description>Total message payload must not exceed MaxPayloadSize (default 228 bytes)</description></item>
/// </list>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// [GpuNativeActor]
/// public interface IPhysicsActor : IGrainWithIntegerKey
/// {
///     [GpuHandler]
///     Task&lt;SimResult&gt; SimulateAsync(SimParams input);
///
///     [GpuHandler(EnableChunking = true)]
///     Task&lt;float[]&gt; ProcessBatchAsync(float[] data);
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class GpuHandlerAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the maximum payload size in bytes for a single message.
    /// Default is 228 bytes (ring kernel message payload limit).
    /// </summary>
    /// <remarks>
    /// The ring kernel control block reserves 28 bytes for header information,
    /// leaving 228 bytes for user payload in a 256-byte message slot.
    /// </remarks>
    public int MaxPayloadSize { get; set; } = 228;

    /// <summary>
    /// Gets or sets whether automatic chunking is enabled for large payloads.
    /// When true, arrays and large structs will be automatically split across multiple messages.
    /// </summary>
    public bool EnableChunking { get; set; }

    /// <summary>
    /// Gets or sets the handler execution mode.
    /// </summary>
    public GpuHandlerMode Mode { get; set; } = GpuHandlerMode.RequestResponse;

    /// <summary>
    /// Gets or sets the queue depth for this handler's message queue.
    /// Default is 1024 messages.
    /// </summary>
    public int QueueDepth { get; set; } = 1024;
}

/// <summary>
/// Specifies the execution mode for a GPU handler method.
/// </summary>
public enum GpuHandlerMode
{
    /// <summary>
    /// Standard request-response pattern. Caller waits for result.
    /// </summary>
    RequestResponse = 0,

    /// <summary>
    /// Fire-and-forget pattern. Caller does not wait for result.
    /// </summary>
    FireAndForget = 1,

    /// <summary>
    /// Streaming pattern. Multiple results returned over time.
    /// </summary>
    Streaming = 2
}
