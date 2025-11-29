namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

/// <summary>
/// Graph node types
/// </summary>
public enum GraphNodeType
{
    /// <summary>
    /// Kernel execution node.
    /// </summary>
    Kernel,

    /// <summary>
    /// Memory copy operation node.
    /// </summary>
    MemCopy,

    /// <summary>
    /// Memory set operation node.
    /// </summary>
    MemSet,

    /// <summary>
    /// Synchronization barrier node.
    /// </summary>
    Barrier,

    /// <summary>
    /// Host callback invocation node.
    /// </summary>
    HostCallback
}