using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Kernels;

/// <summary>
/// Base implementation of IGpuKernel providing common boilerplate.
/// Derive from this class to implement custom kernels.
/// </summary>
public abstract class GpuKernelBase<TIn, TOut> : IGpuKernel<TIn, TOut>
{
    private bool _isInitialized;
    private bool _isDisposed;

    /// <summary>
    /// Kernel unique identifier
    /// </summary>
    public abstract string KernelId { get; }

    /// <summary>
    /// Kernel display name for diagnostics
    /// </summary>
    public virtual string DisplayName => KernelId;

    /// <summary>
    /// Backend provider name ("CPU", "DotCompute", "CUDA", etc.)
    /// </summary>
    public abstract string BackendProvider { get; }

    /// <summary>
    /// Whether kernel is initialized
    /// </summary>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Whether kernel uses GPU acceleration (false for CPU fallback)
    /// </summary>
    public abstract bool IsGpuAccelerated { get; }

    /// <summary>
    /// Initialize kernel resources.
    /// Override to perform custom initialization (e.g., compile GPU code).
    /// </summary>
    public virtual Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        _isInitialized = true;
        return Task.CompletedTask;
    }

    /// <summary>
    /// Execute kernel with single input.
    /// Must be implemented by derived classes.
    /// </summary>
    public abstract Task<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default);

    /// <summary>
    /// Execute kernel with batch input.
    /// Default implementation calls ExecuteAsync for each item (override for batch optimization).
    /// </summary>
    public virtual async Task<TOut[]> ExecuteBatchAsync(TIn[] inputs, CancellationToken cancellationToken = default)
    {
        var results = new TOut[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = await ExecuteAsync(inputs[i], cancellationToken);
        }
        return results;
    }

    /// <summary>
    /// Get estimated execution time for input size.
    /// Override for more accurate estimates based on kernel complexity.
    /// </summary>
    public virtual long GetEstimatedExecutionTimeMicroseconds(int inputSize)
    {
        // Default: Assume 1μs per element (override with kernel-specific logic)
        return inputSize;
    }

    /// <summary>
    /// Get memory requirements for kernel execution.
    /// Override to provide accurate memory estimates.
    /// </summary>
    public virtual KernelMemoryRequirements GetMemoryRequirements()
    {
        // Default: Assume 1KB input + 1KB output + 1KB working
        return new KernelMemoryRequirements(
            InputMemoryBytes: 1024,
            OutputMemoryBytes: 1024,
            WorkingMemoryBytes: 1024,
            TotalMemoryBytes: 3072);
    }

    /// <summary>
    /// Validate input data.
    /// Override for custom validation logic.
    /// </summary>
    public virtual KernelValidationResult ValidateInput(TIn input)
    {
        if (input == null)
            return KernelValidationResult.Invalid("Input cannot be null");

        return KernelValidationResult.Valid();
    }

    /// <summary>
    /// Warmup kernel for optimal performance.
    /// Override to perform custom warmup (e.g., JIT compilation, cache warming).
    /// </summary>
    public virtual Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        // Default: No warmup needed
        return Task.CompletedTask;
    }

    /// <summary>
    /// Dispose kernel resources.
    /// Override to cleanup GPU memory, handles, etc.
    /// </summary>
    public virtual void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
        _isInitialized = false;

        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Ensures kernel is initialized before execution
    /// </summary>
    protected void EnsureInitialized()
    {
        if (!_isInitialized)
            throw new InvalidOperationException(
                $"Kernel '{KernelId}' is not initialized. Call InitializeAsync() first.");
    }

    /// <summary>
    /// Ensures kernel is not disposed
    /// </summary>
    protected void EnsureNotDisposed()
    {
        if (_isDisposed)
            throw new ObjectDisposedException(
                GetType().Name,
                $"Kernel '{KernelId}' has been disposed");
    }
}
