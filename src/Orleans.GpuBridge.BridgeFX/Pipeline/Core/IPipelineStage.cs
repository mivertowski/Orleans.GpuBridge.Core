using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Core;

/// <summary>
/// Interface for pipeline stages.
/// </summary>
public interface IPipelineStage
{
    /// <summary>
    /// Gets the input type for this pipeline stage.
    /// </summary>
    Type InputType { get; }

    /// <summary>
    /// Gets the output type for this pipeline stage.
    /// </summary>
    Type OutputType { get; }

    /// <summary>
    /// Processes the input and returns the transformed output.
    /// </summary>
    /// <param name="input">The input object to process.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The processed output.</returns>
    Task<object?> ProcessAsync(object input, CancellationToken ct = default);
}