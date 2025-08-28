using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Core;

/// <summary>
/// Interface for pipeline stages
/// </summary>
public interface IPipelineStage
{
    Type InputType { get; }
    Type OutputType { get; }
    Task<object?> ProcessAsync(object input, CancellationToken ct = default);
}