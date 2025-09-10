namespace Orleans.GpuBridge.Backends.DotCompute.Interfaces;

/// <summary>
/// Kernel execution handle
/// </summary>
public interface IKernelExecution
{
    bool IsComplete { get; }
    Task WaitForCompletionAsync(CancellationToken ct = default);
    TimeSpan GetExecutionTime();
}