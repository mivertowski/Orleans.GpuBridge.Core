using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Grains.Models;
using Orleans.GpuBridge.Grains.State;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Extension methods for GpuMemoryInfo to provide test-specific properties
/// </summary>
public static class GpuMemoryInfoExtensions
{
    /// <summary>
    /// Gets the total allocated bytes from the GpuMemoryInfo
    /// </summary>
    public static long TotalAllocatedBytes(this GpuMemoryInfo info) => info.AllocatedMemoryBytes;
    
    /// <summary>
    /// Gets the allocation count - calculated from utilization data
    /// </summary>
    public static int AllocationCount(this GpuMemoryInfo info) => 
        (int)(info.AllocatedMemoryBytes > 0 ? Math.Max(1, info.AllocatedMemoryBytes / (1024 * 1024)) : 0);
    
    /// <summary>
    /// Creates a mock allocations dictionary for testing
    /// </summary>
    public static IReadOnlyDictionary<string, GpuMemoryHandle> Allocations(this GpuMemoryInfo info)
    {
        var allocations = new Dictionary<string, GpuMemoryHandle>();
        
        // Create mock allocations based on allocated memory
        if (info.AllocatedMemoryBytes > 0)
        {
            var count = Math.Max(1, info.AllocationCount());
            var sizePerAllocation = info.AllocatedMemoryBytes / count;
            
            for (int i = 0; i < count; i++)
            {
                var handle = GpuMemoryHandle.Create($"test-handle-{i}", sizePerAllocation);
                allocations[handle.Id] = handle;
            }
        }
        
        return allocations;
    }
}