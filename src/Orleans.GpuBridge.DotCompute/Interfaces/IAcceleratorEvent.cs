using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.DotCompute.Interfaces;

/// <summary>
/// Represents an event that can be recorded and waited upon in accelerator streams
/// </summary>
/// <remarks>
/// Events provide a mechanism for synchronization between different streams and
/// for timing GPU operations.
/// </remarks>
public interface IAcceleratorEvent : IDisposable
{
    /// <summary>
    /// Gets the unique identifier for this event
    /// </summary>
    string EventId { get; }
    
    /// <summary>
    /// Gets whether this event has been recorded
    /// </summary>
    bool IsRecorded { get; }
    
    /// <summary>
    /// Gets whether this event has completed
    /// </summary>
    bool IsCompleted { get; }
    
    /// <summary>
    /// Waits for this event to complete
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the wait operation</returns>
    Task WaitAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets the elapsed time between this event and another event
    /// </summary>
    /// <param name="startEvent">The start event</param>
    /// <returns>Elapsed time in milliseconds</returns>
    double GetElapsedTime(IAcceleratorEvent startEvent);
    
    /// <summary>
    /// Queries whether the event has completed without blocking
    /// </summary>
    /// <returns>True if the event has completed, false otherwise</returns>
    bool Query();
}