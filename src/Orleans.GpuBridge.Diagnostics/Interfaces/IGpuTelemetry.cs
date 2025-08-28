using System.Diagnostics;
using Orleans.GpuBridge.Diagnostics.Enums;

namespace Orleans.GpuBridge.Diagnostics.Interfaces;

/// <summary>
/// Interface for GPU telemetry collection and monitoring operations.
/// This service provides comprehensive tracking of GPU performance metrics, kernel execution statistics,
/// memory operations, and system-wide GPU utilization patterns for observability and performance optimization.
/// </summary>
public interface IGpuTelemetry
{
    /// <summary>
    /// Starts a distributed tracing activity for GPU kernel execution.
    /// This creates a parent span that tracks the entire kernel execution lifecycle
    /// including launch, execution, and completion phases.
    /// </summary>
    /// <param name="kernelName">The name of the GPU kernel being executed.</param>
    /// <param name="deviceIndex">The index of the GPU device where the kernel will execute.</param>
    /// <returns>
    /// An <see cref="Activity"/> instance that represents the kernel execution span,
    /// or <c>null</c> if distributed tracing is not enabled or available.
    /// The returned activity should be disposed when kernel execution completes.
    /// </returns>
    /// <example>
    /// <code>
    /// using var activity = telemetry.StartKernelExecution("VectorAdd", 0);
    /// // Perform kernel execution...
    /// telemetry.RecordKernelExecution("VectorAdd", 0, executionTime, success);
    /// </code>
    /// </example>
    Activity? StartKernelExecution(string kernelName, int deviceIndex);

    /// <summary>
    /// Records the completion of a GPU kernel execution with timing and success information.
    /// This method updates performance counters, histograms, and distributed tracing spans
    /// to provide comprehensive execution monitoring.
    /// </summary>
    /// <param name="kernelName">The name of the executed kernel.</param>
    /// <param name="deviceIndex">The index of the GPU device that executed the kernel.</param>
    /// <param name="duration">The total execution time from kernel launch to completion.</param>
    /// <param name="success">Whether the kernel execution completed successfully without errors.</param>
    void RecordKernelExecution(string kernelName, int deviceIndex, TimeSpan duration, bool success);

    /// <summary>
    /// Records GPU memory transfer operations for bandwidth monitoring and optimization.
    /// This tracks data movement between host and device memory, including transfer rates
    /// and directional patterns for performance analysis.
    /// </summary>
    /// <param name="direction">The direction of the memory transfer operation.</param>
    /// <param name="bytes">The number of bytes transferred.</param>
    /// <param name="duration">The time taken to complete the transfer operation.</param>
    void RecordMemoryTransfer(TransferDirection direction, long bytes, TimeSpan duration);

    /// <summary>
    /// Records GPU memory allocation operations for memory usage tracking and leak detection.
    /// This method tracks both successful allocations and failures to monitor memory health
    /// and identify potential memory management issues.
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device where allocation was attempted.</param>
    /// <param name="bytes">The number of bytes requested for allocation.</param>
    /// <param name="success">Whether the allocation operation succeeded.</param>
    void RecordMemoryAllocation(int deviceIndex, long bytes, bool success);

    /// <summary>
    /// Records detailed information about GPU memory allocation failures.
    /// This provides diagnostic information for troubleshooting memory allocation issues
    /// and understanding resource constraints.
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device where allocation failed.</param>
    /// <param name="requestedBytes">The number of bytes that were requested but could not be allocated.</param>
    /// <param name="reason">A descriptive reason for the allocation failure (e.g., "Out of memory", "Fragmentation").</param>
    void RecordAllocationFailure(int deviceIndex, long requestedBytes, string reason);

    /// <summary>
    /// Records the current depth of the GPU work queue for load monitoring.
    /// This helps track work distribution and identify potential bottlenecks
    /// or underutilization scenarios across GPU devices.
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device.</param>
    /// <param name="depth">The number of queued work items waiting for execution.</param>
    void RecordQueueDepth(int deviceIndex, int depth);

    /// <summary>
    /// Records grain activation events for GPU-related grains.
    /// This tracks the lifecycle of GPU-aware Orleans grains and their activation performance,
    /// which is important for understanding grain placement and resource initialization overhead.
    /// </summary>
    /// <param name="grainType">The type name of the grain being activated (e.g., "GpuBatchGrain").</param>
    /// <param name="duration">The time taken to complete the grain activation process.</param>
    void RecordGrainActivation(string grainType, TimeSpan duration);

    /// <summary>
    /// Records the execution of individual pipeline stages in GPU processing pipelines.
    /// This provides detailed performance tracking for complex multi-stage GPU operations
    /// and helps identify bottlenecks within processing pipelines.
    /// </summary>
    /// <param name="stageName">The name of the pipeline stage (e.g., "DataPreprocessing", "KernelExecution", "ResultAggregation").</param>
    /// <param name="duration">The time taken to complete the stage execution.</param>
    /// <param name="success">Whether the stage completed successfully without errors.</param>
    void RecordPipelineStage(string stageName, TimeSpan duration, bool success);
}