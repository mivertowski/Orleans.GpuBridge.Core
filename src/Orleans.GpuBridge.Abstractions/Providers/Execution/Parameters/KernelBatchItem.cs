using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;

/// <summary>
/// Item in a kernel batch
/// </summary>
public sealed record KernelBatchItem(
    CompiledKernel Kernel,
    KernelExecutionParameters Parameters,
    string? ItemId = null);