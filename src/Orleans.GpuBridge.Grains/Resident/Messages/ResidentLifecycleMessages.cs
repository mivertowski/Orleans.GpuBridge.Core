namespace Orleans.GpuBridge.Grains.Resident.Messages;

/// <summary>
/// Message to initialize the Ring Kernel.
/// Sent once at launch.
/// </summary>
public sealed record InitializeMessage : ResidentMessage
{
    /// <summary>
    /// Maximum memory pool size in bytes.
    /// </summary>
    public long MaxPoolSizeBytes { get; init; }

    /// <summary>
    /// Maximum kernel cache size.
    /// </summary>
    public int MaxKernelCacheSize { get; init; }

    /// <summary>
    /// Device index to use.
    /// </summary>
    public int DeviceIndex { get; init; }

    public InitializeMessage(long maxPoolSizeBytes, int maxKernelCacheSize, int deviceIndex)
    {
        MaxPoolSizeBytes = maxPoolSizeBytes;
        MaxKernelCacheSize = maxKernelCacheSize;
        DeviceIndex = deviceIndex;
    }
}

/// <summary>
/// Message to shutdown the Ring Kernel gracefully.
/// </summary>
public sealed record ShutdownMessage : ResidentMessage
{
    /// <summary>
    /// Whether to drain pending messages before shutdown.
    /// </summary>
    public bool DrainPendingMessages { get; init; }

    public ShutdownMessage(bool drainPendingMessages = true)
    {
        DrainPendingMessages = drainPendingMessages;
    }
}
