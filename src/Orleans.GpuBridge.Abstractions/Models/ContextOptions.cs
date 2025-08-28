using System.Collections.Generic;

namespace Orleans.GpuBridge.Abstractions.Models;

/// <summary>
/// Configuration options for creating a compute context.
/// These options control various aspects of context behavior including debugging,
/// profiling, concurrency, and backend-specific customizations.
/// </summary>
/// <param name="EnableProfiling">
/// Indicates whether profiling should be enabled for this context.
/// When enabled, the context will collect timing and performance metrics
/// for kernels and operations executed within the context.
/// This may introduce slight performance overhead but provides valuable
/// debugging and optimization information. Default is false.
/// </param>
/// <param name="EnableDebugMode">
/// Indicates whether debug mode should be enabled for this context.
/// Debug mode typically enables additional validation, error checking,
/// and diagnostic information that can help during development.
/// This may significantly impact performance and should only be used during development.
/// Default is false.
/// </param>
/// <param name="CommandQueueCount">
/// The number of command queues to create for this context.
/// Multiple command queues enable concurrent execution of independent operations
/// and can improve throughput for workloads with multiple parallel streams.
/// Must be at least 1. Default is 1.
/// </param>
/// <param name="EnableOutOfOrderExecution">
/// Indicates whether out-of-order execution should be enabled for the context.
/// When enabled, operations may complete in a different order than they were submitted,
/// potentially improving performance through better resource utilization.
/// Applications must handle proper synchronization when this is enabled.
/// Default is false.
/// </param>
/// <param name="CustomOptions">
/// Backend-specific custom options as key-value pairs.
/// This allows backend providers to accept additional configuration parameters
/// that are not covered by the standard options. The interpretation of these
/// options is specific to each backend implementation.
/// If null, no custom options are applied.
/// </param>
public sealed record ContextOptions(
    bool EnableProfiling = false,
    bool EnableDebugMode = false,
    int CommandQueueCount = 1,
    bool EnableOutOfOrderExecution = false,
    IReadOnlyDictionary<string, object>? CustomOptions = null);