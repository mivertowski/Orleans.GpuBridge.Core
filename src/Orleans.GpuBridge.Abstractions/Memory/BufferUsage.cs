using System;

namespace Orleans.GpuBridge.Abstractions.Memory;

/// <summary>
/// Buffer usage flags
/// </summary>
[Flags]
public enum BufferUsage
{
    None = 0,
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = ReadOnly | WriteOnly,
    Persistent = 4,
    Streaming = 8,
    UnifiedMemory = 16
}