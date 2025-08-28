namespace Orleans.GpuBridge.Abstractions.Kernels;

/// <summary>
/// Result from kernel execution
/// </summary>
public sealed record KernelResult<TOut>(
    IReadOnlyList<TOut> Results,
    TimeSpan ExecutionTime,
    KernelHandle Handle,
    bool Success = true,
    string? Error = null) where TOut : notnull;