// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Generation;

/// <summary>
/// Marks a property in a [GpuNativeActor] interface as GPU-resident state.
/// The source generator will include this property in the generated state struct.
/// </summary>
/// <remarks>
/// <para>
/// Properties marked with this attribute will be:
/// <list type="bullet">
/// <item><description>Combined into a single blittable state struct</description></item>
/// <item><description>Stored in GPU memory alongside the ring kernel</description></item>
/// <item><description>Accessible by the kernel handler between messages</description></item>
/// <item><description>Persisted to Orleans grain state on deactivation</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Requirements:</strong>
/// <list type="bullet">
/// <item><description>Property type must be blittable (value type without references)</description></item>
/// <item><description>Property must have a getter (setter is optional for read-only state)</description></item>
/// </list>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// [GpuNativeActor]
/// public interface ICounterActor : IGrainWithIntegerKey
/// {
///     [GpuState]
///     int Counter { get; }
///
///     [GpuState]
///     long LastUpdateTicks { get; }
///
///     [GpuHandler]
///     Task&lt;int&gt; IncrementAsync(int delta);
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class GpuStateAttribute : Attribute
{
    /// <summary>
    /// Gets or sets whether this state field should be persisted to Orleans storage.
    /// Default is true.
    /// </summary>
    public bool Persist { get; set; } = true;

    /// <summary>
    /// Gets or sets the initial value expression for this state field (C# expression).
    /// If null, the default value for the type is used.
    /// </summary>
    public string? InitialValue { get; set; }
}
