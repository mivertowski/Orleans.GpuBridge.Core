using System.Threading.Tasks;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.Streams;

namespace Orleans.GpuBridge.Grains.Stream;

/// <summary>
/// Interface for stream processing grain
/// </summary>
public interface IGpuStreamGrain<TIn, TOut> : IGrainWithStringKey
    where TIn : notnull
    where TOut : notnull
{
    /// <summary>
    /// Starts processing from input stream to output stream
    /// </summary>
    Task StartProcessingAsync(
        StreamId inputStream,
        StreamId outputStream,
        GpuExecutionHints? hints = null);
    
    /// <summary>
    /// Stops stream processing
    /// </summary>
    Task StopProcessingAsync();
    
    /// <summary>
    /// Gets the current processing status
    /// </summary>
    Task<StreamProcessingStatus> GetStatusAsync();
    
    /// <summary>
    /// Gets processing statistics
    /// </summary>
    Task<StreamProcessingStats> GetStatsAsync();
}