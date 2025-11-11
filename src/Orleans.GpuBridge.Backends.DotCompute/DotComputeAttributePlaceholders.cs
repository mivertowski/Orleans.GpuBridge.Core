// TEMPORARY: Placeholder attributes for DotCompute 0.4.2-rc2 features
// These will be replaced when DotCompute implements the actual attributes
//
// Correct namespace: DotCompute.Generators.Kernel.Attributes
// These are not yet available in the current NuGet package.
//
// TODO: Remove this file once DotCompute 0.4.2-rc2 is released with these features

namespace DotCompute.Generators.Kernel.Attributes;

/// <summary>
/// Placeholder for DotCompute Kernel attribute (will be provided by DotCompute framework).
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
public sealed class KernelAttribute : Attribute
{
    /// <summary>Enables automatic GPU timestamp injection.</summary>
    public bool EnableTimestamps { get; set; }
    /// <summary>Enables barrier synchronization.</summary>
    public bool EnableBarriers { get; set; }
    /// <summary>Barrier synchronization scope.</summary>
    public BarrierScope BarrierScope { get; set; }
    /// <summary>Memory ordering semantics.</summary>
    public MemoryOrderingMode MemoryOrdering { get; set; }
}

/// <summary>
/// Placeholder for DotCompute RingKernel attribute (will be provided by DotCompute framework).
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
public sealed class RingKernelAttribute : Attribute
{
    /// <summary>Size of the message queue.</summary>
    public int MessageQueueSize { get; set; }
    /// <summary>Message processing mode (Continuous/Batch/Adaptive).</summary>
    public RingProcessingMode ProcessingMode { get; set; }
    /// <summary>Enables automatic GPU timestamp injection.</summary>
    public bool EnableTimestamps { get; set; }
    /// <summary>Memory ordering semantics.</summary>
    public MemoryOrderingMode MemoryOrdering { get; set; }
    /// <summary>Maximum messages processed per iteration.</summary>
    public int MaxMessagesPerIteration { get; set; }
    /// <summary>Enables barrier synchronization.</summary>
    public bool EnableBarriers { get; set; }
}

/// <summary>
/// Placeholder for ring kernel processing mode enum.
/// </summary>
public enum RingProcessingMode
{
    /// <summary>Continuous message processing.</summary>
    Continuous,
    /// <summary>Batch message processing.</summary>
    Batch,
    /// <summary>Adaptive message processing.</summary>
    Adaptive
}

/// <summary>
/// Placeholder for memory ordering mode enum.
/// </summary>
public enum MemoryOrderingMode
{
    /// <summary>Relaxed memory ordering.</summary>
    Relaxed,
    /// <summary>Acquire memory ordering.</summary>
    Acquire,
    /// <summary>Release memory ordering.</summary>
    Release,
    /// <summary>Release-Acquire memory ordering.</summary>
    ReleaseAcquire,
    /// <summary>Sequentially consistent memory ordering.</summary>
    SequentiallyConsistent
}

/// <summary>
/// Placeholder for barrier scope enum.
/// </summary>
public enum BarrierScope
{
    /// <summary>Work group barrier scope.</summary>
    WorkGroup,
    /// <summary>Device-wide barrier scope.</summary>
    Device,
    /// <summary>System-wide barrier scope.</summary>
    System
}
