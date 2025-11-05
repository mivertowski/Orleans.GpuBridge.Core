using Orleans.GpuBridge.Backends.DotCompute.Memory;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Kernel launch parameters for GPU execution configuration
/// </summary>
public sealed class KernelLaunchParams
{
    /// <summary>
    /// Gets or sets the total number of work items to execute globally
    /// </summary>
    public int GlobalWorkSize { get; set; }

    /// <summary>
    /// Gets or sets the number of work items in each work group (default: 256)
    /// </summary>
    public int LocalWorkSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the amount of shared memory in bytes available to each work group
    /// </summary>
    public int SharedMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the dictionary of input/output buffers indexed by parameter position
    /// </summary>
    public Dictionary<int, IUnifiedBuffer<byte>> Buffers { get; set; } = new();

    /// <summary>
    /// Gets or sets the dictionary of constant values passed to the kernel by name
    /// </summary>
    public Dictionary<string, object> Constants { get; set; } = new();
}