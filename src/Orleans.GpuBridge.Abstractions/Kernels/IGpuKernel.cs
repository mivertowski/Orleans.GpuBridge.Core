using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Kernels;

/// <summary>
/// Core abstraction for GPU kernel execution.
/// Provides Orleans-integrated kernel lifecycle and execution contract.
/// </summary>
/// <typeparam name="TIn">Input data type</typeparam>
/// <typeparam name="TOut">Output result type</typeparam>
/// <remarks>
/// This abstraction allows consumers to:
/// - Use DotCompute directly for full power (advanced users)
/// - Use Orleans-integrated facade for standard patterns (GpuGrain)
/// - Switch between CPU and GPU backends transparently
/// </remarks>
public interface IGpuKernel<TIn, TOut> : IDisposable
{
    /// <summary>
    /// Unique identifier for this kernel instance
    /// </summary>
    string KernelId { get; }

    /// <summary>
    /// Kernel display name for diagnostics and logging
    /// </summary>
    string DisplayName { get; }

    /// <summary>
    /// Backend provider this kernel is running on
    /// </summary>
    string BackendProvider { get; }

    /// <summary>
    /// Whether this kernel is currently initialized and ready for execution
    /// </summary>
    bool IsInitialized { get; }

    /// <summary>
    /// Whether this kernel is GPU-accelerated or CPU fallback
    /// </summary>
    bool IsGpuAccelerated { get; }

    /// <summary>
    /// Initializes the kernel and prepares GPU resources
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the initialization operation</returns>
    Task InitializeAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes the kernel with single input
    /// </summary>
    /// <param name="input">Input data</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Kernel execution result</returns>
    Task<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes the kernel with batch input for optimal throughput
    /// </summary>
    /// <param name="inputs">Batch of input data</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Batch of kernel execution results</returns>
    Task<TOut[]> ExecuteBatchAsync(TIn[] inputs, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets estimated execution time for given input size
    /// </summary>
    /// <param name="inputSize">Size of input data</param>
    /// <returns>Estimated execution time in microseconds</returns>
    long GetEstimatedExecutionTimeMicroseconds(int inputSize);

    /// <summary>
    /// Gets memory requirements for this kernel
    /// </summary>
    /// <returns>Memory requirements in bytes</returns>
    KernelMemoryRequirements GetMemoryRequirements();

    /// <summary>
    /// Validates that input data meets kernel requirements
    /// </summary>
    /// <param name="input">Input data to validate</param>
    /// <returns>Validation result with error details if invalid</returns>
    KernelValidationResult ValidateInput(TIn input);

    /// <summary>
    /// Warms up the kernel for optimal performance
    /// (JIT compilation, GPU kernel compilation, cache warming)
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the warmup operation</returns>
    Task WarmupAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Memory requirements for kernel execution
/// </summary>
public sealed record KernelMemoryRequirements(
    long InputMemoryBytes,
    long OutputMemoryBytes,
    long WorkingMemoryBytes,
    long TotalMemoryBytes);

/// <summary>
/// Result of kernel input validation
/// </summary>
public sealed record KernelValidationResult(
    bool IsValid,
    string? ErrorMessage = null,
    string[]? ValidationErrors = null)
{
    public static KernelValidationResult Valid() => new(true);
    public static KernelValidationResult Invalid(string errorMessage, params string[] validationErrors) =>
        new(false, errorMessage, validationErrors);
}
