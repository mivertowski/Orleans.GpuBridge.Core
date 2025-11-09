using System;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Interface for fallback executors that can indicate whether to fallback
/// </summary>
public interface IFallbackAware
{
    /// <summary>
    /// Determines if fallback should occur for the given exception
    /// </summary>
    bool ShouldFallback(Exception exception);
}
