namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Configuration options for creating a command queue.
/// These options control the behavior and capabilities of the command queue,
/// including profiling, execution ordering, and priority settings.
/// </summary>
/// <param name="EnableProfiling">
/// Indicates whether profiling should be enabled for operations submitted to this queue.
/// When enabled, the queue will collect detailed timing and performance metrics
/// for each kernel execution and memory operation. This information can be used
/// for performance analysis and optimization but may introduce slight overhead.
/// Default is false.
/// </param>
/// <param name="EnableOutOfOrderExecution">
/// Indicates whether out-of-order execution should be enabled for this queue.
/// When enabled, operations in the queue may complete in a different order
/// than they were submitted, potentially improving throughput through better
/// resource utilization. Applications must use proper synchronization mechanisms
/// (such as barriers) when order dependencies exist. Default is false.
/// </param>
/// <param name="Priority">
/// The priority level for this command queue relative to other queues.
/// Higher values indicate higher priority, which may influence scheduling
/// decisions when multiple queues are competing for device resources.
/// The exact behavior and supported range depends on the backend implementation.
/// Default is 0 (normal priority).
/// </param>
public sealed record CommandQueueOptions(
    bool EnableProfiling = false,
    bool EnableOutOfOrderExecution = false,
    int Priority = 0);