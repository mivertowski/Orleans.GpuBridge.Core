using System;

namespace Orleans.GpuBridge.Abstractions.Memory;

/// <summary>
/// Buffer usage flags
/// </summary>
[Flags]
public enum BufferUsage
{
    /// <summary>
    /// No specific usage flags.
    /// </summary>
    None = 0,

    /// <summary>
    /// Buffer is read-only from the GPU perspective.
    /// </summary>
    ReadOnly = 1,

    /// <summary>
    /// Buffer is write-only from the GPU perspective.
    /// </summary>
    WriteOnly = 2,

    /// <summary>
    /// Buffer supports both read and write operations.
    /// </summary>
    ReadWrite = ReadOnly | WriteOnly,

    /// <summary>
    /// Buffer persists across multiple kernel executions.
    /// </summary>
    Persistent = 4,

    /// <summary>
    /// Buffer is used for streaming data (frequent updates).
    /// </summary>
    Streaming = 8,

    /// <summary>
    /// Buffer uses unified memory accessible from both CPU and GPU.
    /// </summary>
    UnifiedMemory = 16
}