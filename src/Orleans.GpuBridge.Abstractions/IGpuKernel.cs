using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Interface for GPU kernel execution
/// </summary>
public interface IGpuKernel<TIn, TOut> 
    where TIn : notnull 
    where TOut : notnull
{
    /// <summary>
    /// Submits a batch of items for GPU processing
    /// </summary>
    ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default);
    
    /// <summary>
    /// Reads results from a submitted batch
    /// </summary>
    IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        CancellationToken ct = default);
    
    /// <summary>
    /// Gets information about this kernel
    /// </summary>
    ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default);
}

/// <summary>
/// Handle for a submitted kernel execution
/// </summary>
public sealed record KernelHandle(
    string Id,
    DateTimeOffset SubmittedAt,
    KernelStatus Status = KernelStatus.Queued)
{
    public static KernelHandle Create() => new(
        Guid.NewGuid().ToString("N"),
        DateTimeOffset.UtcNow);
}

/// <summary>
/// Status of kernel execution
/// </summary>
public enum KernelStatus
{
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled
}

/// <summary>
/// Information about a kernel
/// </summary>
public sealed record KernelInfo(
    KernelId Id,
    string Description,
    Type InputType,
    Type OutputType,
    bool SupportsGpu,
    int PreferredBatchSize,
    IReadOnlyDictionary<string, object>? Metadata = null);

/// <summary>
/// Result from kernel execution
/// </summary>
public sealed record KernelResult<TOut>(
    IReadOnlyList<TOut> Results,
    TimeSpan ExecutionTime,
    KernelHandle Handle,
    bool Success = true,
    string? Error = null) where TOut : notnull;