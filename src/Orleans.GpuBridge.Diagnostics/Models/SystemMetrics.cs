namespace Orleans.GpuBridge.Diagnostics.Models;

/// <summary>
/// Represents system-level performance metrics for the Orleans.GpuBridge host process.
/// </summary>
/// <remarks>
/// This class captures resource consumption metrics for the current process, providing
/// insights into CPU usage, memory consumption, and operating system resource utilization.
/// These metrics are useful for performance monitoring, capacity planning, and troubleshooting.
/// </remarks>
public class SystemMetrics
{
    /// <summary>
    /// Gets or sets the CPU utilization percentage for the current process.
    /// </summary>
    /// <value>
    /// Process CPU usage as a percentage (0.0 to 100.0 per CPU core). Values may exceed
    /// 100% on multi-core systems when the process uses multiple cores simultaneously.
    /// </value>
    /// <remarks>
    /// <para>
    /// CPU utilization represents the fraction of CPU time consumed by the Orleans.GpuBridge
    /// process during the measurement period. This includes:
    /// - GPU kernel preparation and scheduling
    /// - Data marshalling and transfer operations  
    /// - Orleans grain processing overhead
    /// - Background metric collection activities
    /// </para>
    /// <para>
    /// High CPU usage may indicate:
    /// - CPU-intensive GPU kernel preparation
    /// - Inefficient data serialization
    /// - Excessive logging or telemetry overhead
    /// - Orleans grain processing bottlenecks
    /// </para>
    /// <para>
    /// Note: Current implementation returns 0 as a placeholder. Proper CPU usage
    /// calculation requires tracking process CPU time over intervals.
    /// </para>
    /// </remarks>
    public double ProcessCpuUsage { get; set; }

    /// <summary>
    /// Gets or sets the working set memory usage for the current process.
    /// </summary>
    /// <value>
    /// Process memory usage in megabytes (MB). This represents the physical memory
    /// currently allocated to the process by the operating system.
    /// </value>
    /// <remarks>
    /// <para>
    /// Working set memory includes:
    /// - Orleans grain state and object instances
    /// - GPU buffer allocations and staging memory
    /// - Cached kernel compilation artifacts
    /// - Framework and runtime overhead
    /// - Managed heap and native memory allocations
    /// </para>
    /// <para>
    /// Memory growth patterns can indicate:
    /// - Memory leaks in GPU resource management
    /// - Excessive grain state caching
    /// - Large dataset processing without proper cleanup
    /// - Kernel compilation artifact accumulation
    /// </para>
    /// <para>
    /// This metric reflects actual physical memory usage and may differ from
    /// virtual memory allocations reported by other tools.
    /// </para>
    /// </remarks>
    public long ProcessMemoryMB { get; set; }

    /// <summary>
    /// Gets or sets the number of threads currently active in the process.
    /// </summary>
    /// <value>
    /// The total count of threads including main application threads, background
    /// worker threads, and framework-managed threads.
    /// </value>
    /// <remarks>
    /// <para>
    /// Thread count includes:
    /// - Orleans scheduler and grain activation threads
    /// - GPU operation background threads
    /// - Metrics collection and telemetry threads
    /// - .NET runtime threads (GC, thread pool, finalizer)
    /// - Platform-specific system threads
    /// </para>
    /// <para>
    /// Thread count monitoring helps identify:
    /// - Thread pool exhaustion or saturation
    /// - Resource leaks in concurrent operations
    /// - Scaling behavior under load
    /// - Background task accumulation
    /// </para>
    /// <para>
    /// Typical thread counts vary by workload but sudden increases may indicate
    /// resource leaks or excessive concurrent operations.
    /// </para>
    /// </remarks>
    public int ThreadCount { get; set; }

    /// <summary>
    /// Gets or sets the number of operating system handles held by the process.
    /// </summary>
    /// <value>
    /// The total count of OS handles including files, synchronization objects,
    /// registry keys, and other system resources.
    /// </value>
    /// <remarks>
    /// <para>
    /// Handle count includes:
    /// - GPU device handles and driver resources
    /// - File handles for logs, configuration, and temporary files
    /// - Network connection handles
    /// - Synchronization primitive handles (mutexes, events, semaphores)
    /// - Registry and process handles
    /// </para>
    /// <para>
    /// Handle monitoring is important for:
    /// - Resource leak detection
    /// - System resource limit management
    /// - GPU driver resource tracking
    /// - File descriptor exhaustion prevention
    /// </para>
    /// <para>
    /// Windows systems typically limit handles to several thousand per process.
    /// Gradual handle count increases often indicate resource cleanup issues.
    /// </para>
    /// </remarks>
    public int HandleCount { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when these system metrics were collected.
    /// </summary>
    /// <value>
    /// UTC timestamp indicating when the system metrics were captured.
    /// </value>
    /// <remarks>
    /// This timestamp enables metric correlation with GPU device metrics and can be
    /// used for trend analysis, rate calculations, and temporal synchronization of
    /// monitoring data across different metric sources.
    /// </remarks>
    public DateTimeOffset Timestamp { get; set; }
}