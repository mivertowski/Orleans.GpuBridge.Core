namespace Orleans.GpuBridge.DotCompute.Execution;

/// <summary>
/// Kernel execution handle
/// </summary>
public interface IKernelExecution
{
    bool IsComplete { get; }
    Task WaitForCompletionAsync(CancellationToken ct = default);
    TimeSpan GetExecutionTime();
}