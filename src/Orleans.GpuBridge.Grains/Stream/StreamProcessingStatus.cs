using Orleans;

namespace Orleans.GpuBridge.Grains.Stream;

/// <summary>
/// Stream processing status
/// </summary>
[GenerateSerializer]
public enum StreamProcessingStatus
{
    Idle,
    Starting,
    Processing,
    Stopping,
    Stopped,
    Failed
}