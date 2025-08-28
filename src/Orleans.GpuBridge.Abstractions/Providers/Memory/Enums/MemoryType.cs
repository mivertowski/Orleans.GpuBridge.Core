namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;

/// <summary>
/// Types of memory allocation
/// </summary>
public enum MemoryType
{
    /// <summary>Device-only memory</summary>
    Device,
    /// <summary>Host-visible device memory</summary>
    HostVisible,
    /// <summary>Shared memory</summary>
    Shared,
    /// <summary>Constant/uniform memory</summary>
    Constant,
    /// <summary>Texture memory</summary>
    Texture
}