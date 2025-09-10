namespace Orleans.GpuBridge.Abstractions.Domain.ValueObjects;

/// <summary>
/// CUDA compute capability
/// </summary>
public sealed record ComputeCapability(int Major, int Minor)
{
    public override string ToString() => $"{Major}.{Minor}";
    
    /// <summary>
    /// Checks if this compute capability is at least the specified version
    /// </summary>
    public bool IsAtLeast(int major, int minor) => 
        Major > major || (Major == major && Minor >= minor);
}