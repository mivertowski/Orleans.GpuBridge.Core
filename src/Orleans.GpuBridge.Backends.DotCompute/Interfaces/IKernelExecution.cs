namespace Orleans.GpuBridge.Backends.DotCompute.Interfaces;

/// <summary>
/// Kernel execution handle for tracking asynchronous GPU operations
/// </summary>
public interface IKernelExecution
{
    /// <summary>
    /// Gets a value indicating whether the kernel execution has completed
    /// </summary>
    bool IsComplete { get; }

    /// <summary>
    /// Asynchronously waits for the kernel execution to complete
    /// </summary>
    /// <param name="ct">Cancellation token for the wait operation</param>
    /// <returns>A task that completes when kernel execution finishes</returns>
    Task WaitForCompletionAsync(CancellationToken ct = default);

    /// <summary>
    /// Gets the total execution time of the kernel operation
    /// </summary>
    /// <returns>The elapsed time from kernel launch to completion</returns>
    TimeSpan GetExecutionTime();
}