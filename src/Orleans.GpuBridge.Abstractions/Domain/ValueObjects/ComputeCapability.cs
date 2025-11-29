namespace Orleans.GpuBridge.Abstractions.Domain.ValueObjects;

/// <summary>
/// CUDA compute capability
/// </summary>
public sealed record ComputeCapability(int Major, int Minor)
{
    /// <summary>
    /// Returns a string representation of the compute capability in the format "Major.Minor".
    /// </summary>
    /// <returns>A string like "7.5" for compute capability 7.5.</returns>
    public override string ToString() => $"{Major}.{Minor}";

    /// <summary>
    /// Checks if this compute capability is at least the specified version
    /// </summary>
    public bool IsAtLeast(int major, int minor) =>
        Major > major || (Major == major && Minor >= minor);
}