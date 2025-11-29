namespace Orleans.GpuBridge.Abstractions.Memory;

/// <summary>
/// Memory pool for GPU allocations
/// </summary>
public interface IGpuMemoryPool<T> where T : unmanaged
{
    /// <summary>
    /// Rents memory from the pool
    /// </summary>
    IGpuMemory<T> Rent(int minimumLength);

    /// <summary>
    /// Returns memory to the pool
    /// </summary>
    void Return(IGpuMemory<T> memory);

    /// <summary>
    /// Gets pool statistics
    /// </summary>
    MemoryPoolStats GetStats();
}