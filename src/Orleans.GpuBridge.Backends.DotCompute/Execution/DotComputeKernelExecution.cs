using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using OrleansCompiledKernel = Orleans.GpuBridge.Abstractions.Models.CompiledKernel;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// DotCompute kernel execution wrapper for non-blocking execution
/// </summary>
internal sealed class DotComputeKernelExecution : IKernelExecution
{
    private readonly OrleansCompiledKernel _kernel;
    private readonly DotComputeKernelExecutor _executor;
    private readonly ILogger _logger;
    private readonly TaskCompletionSource<KernelExecutionResult> _completionSource;
    private readonly CancellationTokenSource _cancellationSource;
    private volatile bool _isCompleted;
    private volatile bool _isCanceled;
    private volatile KernelExecutionStatus _status = KernelExecutionStatus.Queued;
    private double _progress = 0.0;

    public string ExecutionId { get; }
    public bool IsCompleted => _isCompleted;
    public bool IsComplete => _isCompleted;
    public bool IsCanceled => _isCanceled;
    public OrleansCompiledKernel Kernel => _kernel;
    public KernelExecutionStatus Status => _status;
    public double Progress => _progress;

    public DotComputeKernelExecution(
        string executionId,
        OrleansCompiledKernel kernel,
        DotComputeKernelExecutor executor,
        ILogger logger)
    {
        ExecutionId = executionId ?? throw new ArgumentNullException(nameof(executionId));
        _kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _completionSource = new TaskCompletionSource<KernelExecutionResult>();
        _cancellationSource = new CancellationTokenSource();
    }

    /// <summary>
    /// Waits for the kernel execution to complete
    /// </summary>
    public async Task<KernelExecutionResult> WaitForCompletionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var combinedCts = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, _cancellationSource.Token);

            return await _completionSource.Task.WaitAsync(combinedCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (OperationCanceledException)
        {
            // Internal cancellation
            return new KernelExecutionResult(false, "Kernel execution was canceled");
        }
    }

    /// <summary>
    /// Cancels the kernel execution
    /// </summary>
    public Task CancelAsync()
    {
        if (_isCompleted)
            return Task.CompletedTask;

        try
        {
            _logger.LogDebug("Cancelling DotCompute kernel execution: {ExecutionId}", ExecutionId);
            _isCanceled = true;
            _status = KernelExecutionStatus.Cancelled;
            _cancellationSource.Cancel();
            _completionSource.TrySetCanceled();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cancelling DotCompute kernel execution: {ExecutionId}", ExecutionId);
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// Gets timing information for the execution
    /// </summary>
    public KernelTiming GetTiming()
    {
        return new KernelTiming(TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero, 0L, 0.0);
    }

    /// <summary>
    /// Sets the execution result (internal method)
    /// </summary>
    internal void SetResult(KernelExecutionResult result)
    {
        if (_isCompleted)
            return;

        _isCompleted = true;
        _completionSource.TrySetResult(result);
    }

    /// <summary>
    /// Sets an execution error (internal method)
    /// </summary>
    internal void SetError(Exception exception)
    {
        if (_isCompleted)
            return;

        _isCompleted = true;
        _completionSource.TrySetException(exception);
    }

    public void Dispose()
    {
        if (!_isCompleted && !_isCanceled)
        {
            _cancellationSource?.Cancel();
        }
        _cancellationSource?.Dispose();
        _executor.RemoveActiveExecution(ExecutionId);
    }
}
