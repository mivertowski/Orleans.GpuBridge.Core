using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Serialization;

/// <summary>
/// Compression levels for buffer serialization
/// </summary>
public enum CompressionLevel
{
    /// <summary>
    /// No compression (fastest, largest size)
    /// </summary>
    None,

    /// <summary>
    /// Fastest compression with moderate size reduction
    /// </summary>
    Fastest,

    /// <summary>
    /// Balanced compression optimizing both speed and size
    /// </summary>
    Optimal,

    /// <summary>
    /// Maximum compression (slowest, smallest size)
    /// </summary>
    Maximum
}
