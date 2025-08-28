namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

/// <summary>
/// Graph node types
/// </summary>
public enum GraphNodeType
{
    Kernel,
    MemCopy,
    MemSet,
    Barrier,
    HostCallback
}