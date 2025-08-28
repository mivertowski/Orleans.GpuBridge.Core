namespace Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;

/// <summary>
/// Options for batch execution
/// </summary>
public sealed record BatchExecutionOptions(
    bool ExecuteInParallel = false,
    int MaxParallelism = 4,
    bool StopOnFirstError = true,
    bool EnableProfiling = false);