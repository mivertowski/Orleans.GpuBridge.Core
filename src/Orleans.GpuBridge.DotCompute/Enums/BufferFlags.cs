using System;

namespace Orleans.GpuBridge.DotCompute.Enums;

/// <summary>
/// Buffer allocation flags
/// </summary>
[Flags]
public enum BufferFlags
{
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = ReadOnly | WriteOnly,
    HostVisible = 4,
    DeviceLocal = 8,
    Pinned = 16
}