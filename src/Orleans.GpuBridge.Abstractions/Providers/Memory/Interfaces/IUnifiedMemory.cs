using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;

namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

/// <summary>
/// Unified memory accessible from both host and device
/// </summary>
public interface IUnifiedMemory : IDeviceMemory
{
    /// <summary>
    /// Host pointer to the unified memory
    /// </summary>
    IntPtr HostPointer { get; }
    
    /// <summary>
    /// Prefetches the memory to a specific device
    /// </summary>
    Task PrefetchAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Advises the runtime about memory usage patterns
    /// </summary>
    Task AdviseAsync(
        MemoryAdvice advice,
        IComputeDevice? device = null,
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets a span view of the memory from the host
    /// </summary>
    Span<byte> AsHostSpan();
}