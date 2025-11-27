// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Generation;

/// <summary>
/// Marks a GPU-native actor interface as requiring temporal ordering guarantees.
/// The source generator will inject clock synchronization into generated kernel code.
/// </summary>
/// <remarks>
/// <para>
/// Temporal ordering provides:
/// <list type="bullet">
/// <item><description>Hybrid Logical Clock (HLC) for wall-clock approximation with causality</description></item>
/// <item><description>Vector Clocks for multi-actor causal ordering</description></item>
/// <item><description>Automatic timestamp injection into message headers</description></item>
/// <item><description>GPU-side clock update operations in kernel handlers</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Performance Characteristics:</strong>
/// <list type="bullet">
/// <item><description>HLC update: ~20ns on GPU vs ~50ns on CPU</description></item>
/// <item><description>Vector clock merge: ~50ns for typical sizes</description></item>
/// <item><description>Memory overhead: 16 bytes (HLC) or 8*N bytes (Vector Clock)</description></item>
/// </list>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// [GpuNativeActor]
/// [TemporalOrdered(ClockType = TemporalClockType.HLC)]
/// public interface IAuditActor : IGrainWithIntegerKey
/// {
///     [GpuHandler]
///     Task RecordEventAsync(AuditEvent event);
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Interface, AllowMultiple = false, Inherited = true)]
public sealed class TemporalOrderedAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the type of clock to use for temporal ordering.
    /// Default is HLC (Hybrid Logical Clock).
    /// </summary>
    public TemporalClockType ClockType { get; set; } = TemporalClockType.HLC;

    /// <summary>
    /// Gets or sets whether to validate causal ordering at runtime.
    /// When true, out-of-order messages will be rejected or reordered.
    /// Default is false (best effort ordering).
    /// </summary>
    public bool StrictOrdering { get; set; }

    /// <summary>
    /// Gets or sets the maximum clock drift in milliseconds before triggering recalibration.
    /// Default is 100ms.
    /// </summary>
    public int MaxClockDriftMs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum vector clock size for multi-actor tracking.
    /// Only applies when ClockType is VectorClock.
    /// Default is 16 entries.
    /// </summary>
    public int MaxVectorClockSize { get; set; } = 16;
}

/// <summary>
/// Specifies the type of clock used for temporal ordering.
/// </summary>
public enum TemporalClockType
{
    /// <summary>
    /// Hybrid Logical Clock - combines wall clock with logical counter.
    /// Provides both wall-clock approximation and causal ordering.
    /// Recommended for most use cases.
    /// </summary>
    HLC = 0,

    /// <summary>
    /// Vector Clock - tracks causal dependencies across multiple actors.
    /// Provides stronger ordering guarantees but with higher overhead.
    /// Use when you need to track causality across many actors.
    /// </summary>
    VectorClock = 1,

    /// <summary>
    /// Lamport Clock - simple logical clock with happens-before semantics.
    /// Lower overhead but no wall-clock approximation.
    /// </summary>
    Lamport = 2
}
