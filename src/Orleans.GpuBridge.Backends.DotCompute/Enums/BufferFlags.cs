using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Enums;

/// <summary>
/// Buffer allocation flags for controlling memory access and visibility
/// </summary>
[Flags]
public enum BufferFlags
{
    /// <summary>
    /// Buffer is read-only (can only read from GPU)
    /// </summary>
    ReadOnly = 1,

    /// <summary>
    /// Buffer is write-only (can only write to GPU)
    /// </summary>
    WriteOnly = 2,

    /// <summary>
    /// Buffer supports both read and write operations
    /// </summary>
    ReadWrite = ReadOnly | WriteOnly,

    /// <summary>
    /// Buffer is visible to host CPU for direct access
    /// </summary>
    HostVisible = 4,

    /// <summary>
    /// Buffer resides in device-local memory for optimal GPU performance
    /// </summary>
    DeviceLocal = 8,

    /// <summary>
    /// Buffer memory is pinned (locked) in physical memory to prevent paging
    /// </summary>
    Pinned = 16
}