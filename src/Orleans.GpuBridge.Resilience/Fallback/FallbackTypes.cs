// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Exceptions;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Fallback execution levels
/// </summary>
public enum FallbackLevel
{
    /// <summary>
    /// Optimal GPU execution
    /// </summary>
    Optimal = 0,

    /// <summary>
    /// Reduced GPU resources
    /// </summary>
    Reduced = 1,

    /// <summary>
    /// CPU execution
    /// </summary>
    Degraded = 2,

    /// <summary>
    /// Error/failure state
    /// </summary>
    Failed = 3
}

/// <summary>
/// Represents a single fallback attempt
/// </summary>
public readonly record struct FallbackAttempt(
    FallbackLevel Level,
    string ExecutorType,
    bool Success,
    TimeSpan Duration,
    Exception? Exception);

/// <summary>
/// Exception thrown when all fallback executors fail
/// </summary>
[Serializable]
public sealed class FallbackChainExhaustedException : GpuBridgeException
{
    /// <summary>
    /// The default error code for fallback chain exhaustion.
    /// </summary>
    public const string DefaultErrorCode = "FALLBACK_CHAIN_EXHAUSTED";

    /// <summary>
    /// Gets the list of fallback attempts that were made before exhaustion.
    /// </summary>
    public IReadOnlyList<FallbackAttempt> Attempts { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="FallbackChainExhaustedException"/> class.
    /// </summary>
    /// <param name="operationName">The name of the operation that failed.</param>
    /// <param name="attempts">The list of fallback attempts that were made.</param>
    /// <param name="innerException">The optional inner exception that caused the failure.</param>
    public FallbackChainExhaustedException(
        string operationName,
        IReadOnlyList<FallbackAttempt> attempts,
        Exception? innerException = null)
        : base(DefaultErrorCode,
               $"All fallback executors failed for operation '{operationName}'. Attempts: {attempts.Count}",
               operationName,
               attempts)
    {
        Attempts = attempts;
    }
}

/// <summary>
/// Configuration for fallback chain behavior
/// </summary>
public sealed class FallbackChainOptions
{
    /// <summary>
    /// The configuration section name for fallback chain options.
    /// </summary>
    public const string SectionName = "FallbackChain";

    /// <summary>
    /// Whether to enable automatic degradation
    /// </summary>
    public bool AutoDegradationEnabled { get; set; } = true;

    /// <summary>
    /// Whether to enable automatic recovery
    /// </summary>
    public bool AutoRecoveryEnabled { get; set; } = true;

    /// <summary>
    /// Error rate threshold for degradation (0.0 to 1.0)
    /// </summary>
    public double DegradationErrorThreshold { get; set; } = 0.5;

    /// <summary>
    /// Error rate threshold for recovery (0.0 to 1.0)
    /// </summary>
    public double RecoveryErrorThreshold { get; set; } = 0.1;

    /// <summary>
    /// Minimum requests before considering degradation
    /// </summary>
    public int MinimumRequestsForDegradation { get; set; } = 10;

    /// <summary>
    /// Minimum requests before considering recovery
    /// </summary>
    public int MinimumRequestsForRecovery { get; set; } = 5;

    /// <summary>
    /// Minimum interval between recovery attempts
    /// </summary>
    public TimeSpan MinimumRecoveryInterval { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Time window for error rate calculation
    /// </summary>
    public TimeSpan ErrorRateWindow { get; set; } = TimeSpan.FromMinutes(10);
}
