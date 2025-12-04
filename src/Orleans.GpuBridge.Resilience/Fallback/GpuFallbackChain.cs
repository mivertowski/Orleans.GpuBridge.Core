using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Exceptions;
using Orleans.GpuBridge.Resilience.Policies;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Implements a comprehensive fallback chain: GPU -> CPU -> Error with degradation tracking
/// </summary>
public sealed class GpuFallbackChain<TIn, TOut> : IDisposable
    where TIn : notnull
    where TOut : notnull
{
    private readonly ILogger<GpuFallbackChain<TIn, TOut>> _logger;
    private readonly GpuResiliencePolicy _resiliencePolicy;
    private readonly FallbackChainOptions _options;
    private readonly List<IFallbackExecutor<TIn, TOut>> _executors;
    private readonly FallbackMetricsCollector _metricsCollector;
    private readonly SemaphoreSlim _degradationLock;
    private FallbackLevel _currentLevel;
    private DateTimeOffset _lastDegradation;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuFallbackChain{TIn, TOut}"/> class.
    /// </summary>
    /// <param name="logger">The logger instance for fallback chain logging.</param>
    /// <param name="resiliencePolicy">The GPU resilience policy to apply.</param>
    /// <param name="options">The fallback chain configuration options.</param>
    /// <param name="executors">The collection of fallback executors in priority order.</param>
    /// <exception cref="ArgumentNullException">Thrown when any required parameter is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when no executors are provided.</exception>
    public GpuFallbackChain(
        ILogger<GpuFallbackChain<TIn, TOut>> logger,
        GpuResiliencePolicy resiliencePolicy,
        IOptions<FallbackChainOptions> options,
        IEnumerable<IFallbackExecutor<TIn, TOut>> executors)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _resiliencePolicy = resiliencePolicy ?? throw new ArgumentNullException(nameof(resiliencePolicy));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        _executors = executors?.OrderBy(e => e.Priority).ToList() ?? throw new ArgumentNullException(nameof(executors));

        _metricsCollector = new FallbackMetricsCollector();
        _degradationLock = new SemaphoreSlim(1, 1);
        _currentLevel = FallbackLevel.Optimal;
        _lastDegradation = DateTimeOffset.MinValue;

        if (!_executors.Any())
        {
            throw new InvalidOperationException("At least one fallback executor must be provided");
        }

        _logger.LogInformation("Fallback chain initialized with {Count} executors", _executors.Count);
    }

    /// <summary>
    /// Executes the operation through the fallback chain
    /// </summary>
    public async Task<TOut> ExecuteAsync(
        TIn input,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        var startTime = DateTimeOffset.UtcNow;
        var attempts = new List<FallbackAttempt>();

        try
        {
            // Determine starting executor based on current degradation level
            var startIndex = Math.Max(0, (int)_currentLevel);

            for (int i = startIndex; i < _executors.Count; i++)
            {
                var executor = _executors[i];
                var attemptStart = DateTimeOffset.UtcNow;

                try
                {
                    _logger.LogDebug(
                        "Attempting execution with {ExecutorType} (level {Level}): {OperationName}",
                        executor.GetType().Name, executor.Level, operationName);

                    var result = await executor.ExecuteAsync(input, operationName, cancellationToken);

                    var attempt = new FallbackAttempt(
                        Level: executor.Level,
                        ExecutorType: executor.GetType().Name,
                        Success: true,
                        Duration: DateTimeOffset.UtcNow - attemptStart,
                        Exception: null);

                    attempts.Add(attempt);

                    // Record successful execution
                    _metricsCollector.RecordSuccess(executor.Level, DateTimeOffset.UtcNow - startTime);

                    // Consider recovery if we're degraded and succeeded with a lower-priority executor
                    if (_currentLevel > FallbackLevel.Optimal && executor.Level < _currentLevel)
                    {
                        _ = Task.Run(async () => await ConsiderRecoveryAsync(), cancellationToken);
                    }

                    _logger.LogDebug(
                        "Execution succeeded with {ExecutorType}: {OperationName}",
                        executor.GetType().Name, operationName);

                    return result;
                }
                catch (Exception ex) when (ShouldFallback(ex, executor))
                {
                    var attempt = new FallbackAttempt(
                        Level: executor.Level,
                        ExecutorType: executor.GetType().Name,
                        Success: false,
                        Duration: DateTimeOffset.UtcNow - attemptStart,
                        Exception: ex);

                    attempts.Add(attempt);

                    // Record failure
                    _metricsCollector.RecordFailure(executor.Level, ex);

                    _logger.LogWarning(ex,
                        "Execution failed with {ExecutorType}, falling back to next executor: {OperationName}",
                        executor.GetType().Name, operationName);

                    // Update degradation level if needed
                    await UpdateDegradationLevelAsync(executor.Level);

                    // Continue to next executor
                    continue;
                }
            }

            // All executors failed
            var totalDuration = DateTimeOffset.UtcNow - startTime;
            var lastAttempt = attempts.LastOrDefault();
            var lastException = lastAttempt.Exception ??
                new GpuOperationException(operationName, "All fallback executors failed");

            _metricsCollector.RecordTotalFailure(totalDuration);

            _logger.LogError(lastException,
                "All fallback executors failed for operation: {OperationName}. Attempts: {AttemptCount}",
                operationName, attempts.Count);

            throw new FallbackChainExhaustedException(operationName, attempts, lastException);
        }
        finally
        {
            // Log execution summary
            var totalDuration = DateTimeOffset.UtcNow - startTime;
            _logger.LogInformation(
                "Fallback chain execution completed: {OperationName} in {Duration}ms with {AttemptCount} attempts",
                operationName, totalDuration.TotalMilliseconds, attempts.Count);
        }
    }

    /// <summary>
    /// Gets current fallback metrics
    /// </summary>
    public FallbackChainMetrics GetMetrics()
    {
        return _metricsCollector.GetMetrics(_currentLevel);
    }

    /// <summary>
    /// Forces degradation to a specific level (for testing/manual override)
    /// </summary>
    public async Task ForceDegradationAsync(FallbackLevel level, string reason = "Manual override")
    {
        await _degradationLock.WaitAsync();
        try
        {
            var previousLevel = _currentLevel;
            _currentLevel = level;
            _lastDegradation = DateTimeOffset.UtcNow;

            _logger.LogWarning(
                "Fallback chain degraded from {PreviousLevel} to {NewLevel}: {Reason}",
                previousLevel, level, reason);

            _metricsCollector.RecordDegradation(previousLevel, level, reason);
        }
        finally
        {
            _degradationLock.Release();
        }
    }

    /// <summary>
    /// Determines if we should fallback to the next executor
    /// </summary>
    private bool ShouldFallback(Exception exception, IFallbackExecutor<TIn, TOut> currentExecutor)
    {
        // Always fallback for certain critical exceptions
        if (exception is GpuDeviceException or GpuKernelException or GpuMemoryException)
        {
            return true;
        }

        // Check if executor indicates it should fallback
        if (currentExecutor is IFallbackAware fallbackAware)
        {
            return fallbackAware.ShouldFallback(exception);
        }

        // Default behavior: fallback on most exceptions except cancellation
        return exception is not OperationCanceledException;
    }

    /// <summary>
    /// Updates the degradation level based on failure patterns
    /// </summary>
    private async Task UpdateDegradationLevelAsync(FallbackLevel failedLevel)
    {
        if (!_options.AutoDegradationEnabled) return;

        await _degradationLock.WaitAsync();
        try
        {
            var metrics = _metricsCollector.GetLevelMetrics(failedLevel);

            // Check if we should degrade based on error rate
            if (metrics.ErrorRate > _options.DegradationErrorThreshold &&
                metrics.TotalRequests >= _options.MinimumRequestsForDegradation)
            {
                var newLevel = Math.Max((int)failedLevel + 1, (int)_currentLevel);
                if (newLevel < _executors.Count && newLevel > (int)_currentLevel)
                {
                    var previousLevel = _currentLevel;
                    _currentLevel = (FallbackLevel)newLevel;
                    _lastDegradation = DateTimeOffset.UtcNow;

                    _logger.LogWarning(
                        "Auto-degrading fallback chain from {PreviousLevel} to {NewLevel} due to high error rate: {ErrorRate:P}",
                        previousLevel, _currentLevel, metrics.ErrorRate);

                    _metricsCollector.RecordDegradation(previousLevel, _currentLevel,
                        $"Auto-degradation: {metrics.ErrorRate:P} error rate");
                }
            }
        }
        finally
        {
            _degradationLock.Release();
        }
    }

    /// <summary>
    /// Considers recovery to a better fallback level
    /// </summary>
    private async Task ConsiderRecoveryAsync()
    {
        if (!_options.AutoRecoveryEnabled) return;

        // Don't recover too frequently
        if (DateTimeOffset.UtcNow - _lastDegradation < _options.MinimumRecoveryInterval)
        {
            return;
        }

        await _degradationLock.WaitAsync();
        try
        {
            // Try to recover one level at a time
            var targetLevel = Math.Max((int)_currentLevel - 1, (int)FallbackLevel.Optimal);
            if (targetLevel < (int)_currentLevel)
            {
                var targetLevelEnum = (FallbackLevel)targetLevel;
                var metrics = _metricsCollector.GetLevelMetrics(targetLevelEnum);

                // Check if the higher-priority executor looks healthy
                if (metrics.ErrorRate < _options.RecoveryErrorThreshold ||
                    metrics.TotalRequests < _options.MinimumRequestsForRecovery)
                {
                    var previousLevel = _currentLevel;
                    _currentLevel = targetLevelEnum;

                    _logger.LogInformation(
                        "Recovered fallback chain from {PreviousLevel} to {NewLevel}",
                        previousLevel, _currentLevel);

                    _metricsCollector.RecordRecovery(previousLevel, _currentLevel);
                }
            }
        }
        finally
        {
            _degradationLock.Release();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;
        _degradationLock?.Dispose();
        _metricsCollector?.Dispose();

        foreach (var executor in _executors.OfType<IDisposable>())
        {
            executor.Dispose();
        }

        GC.SuppressFinalize(this);
    }
}
