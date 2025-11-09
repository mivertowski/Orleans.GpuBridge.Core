using System.Threading.Tasks;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Base work item for vectorized kernel execution queue
/// </summary>
internal abstract record WorkItem
{
    /// <summary>
    /// Completion source to signal work item completion or failure
    /// </summary>
    public TaskCompletionSource<bool> CompletionSource { get; set; } = null!;
}

/// <summary>
/// Work item for vectorized addition operation
/// </summary>
internal sealed record VectorAddWorkItem(float[] A, float[] B, float[] Result) : WorkItem;

/// <summary>
/// Work item for fused multiply-add operation
/// </summary>
internal sealed record FmaWorkItem(float[] A, float[] B, float[] C, float[] Result) : WorkItem;

/// <summary>
/// Work item for cache-optimized matrix multiplication
/// </summary>
internal sealed record MatrixMultiplyWorkItem(
    float[] A,
    float[] B,
    float[] Result,
    int M,
    int N,
    int K) : WorkItem;

/// <summary>
/// Work item for vectorized reduction operation
/// </summary>
internal sealed record ReductionWorkItem(
    float[] Input,
    float[] Result,
    ReductionOperation Operation) : WorkItem;
