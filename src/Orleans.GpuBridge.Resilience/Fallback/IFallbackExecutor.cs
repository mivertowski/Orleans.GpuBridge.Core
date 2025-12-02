// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Interface for fallback executors.
/// </summary>
/// <typeparam name="TIn">The input type.</typeparam>
/// <typeparam name="TOut">The output type.</typeparam>
public interface IFallbackExecutor<in TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    /// <summary>
    /// Gets the fallback level of this executor.
    /// </summary>
    FallbackLevel Level { get; }

    /// <summary>
    /// Gets the priority of this executor (lower = higher priority).
    /// </summary>
    int Priority { get; }

    /// <summary>
    /// Executes the operation.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="operationName">The operation name.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The output result.</returns>
    Task<TOut> ExecuteAsync(TIn input, string operationName, CancellationToken cancellationToken);
}
