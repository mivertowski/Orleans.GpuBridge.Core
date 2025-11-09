using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Grains.Resident.Messages;

/// <summary>
/// Message to execute a kernel on GPU with resident memory.
/// </summary>
public sealed record ComputeMessage : ResidentMessage
{
    /// <summary>
    /// Kernel identifier (from KernelId).
    /// </summary>
    public string KernelId { get; init; }

    /// <summary>
    /// Input memory allocation ID.
    /// </summary>
    public string InputAllocationId { get; init; }

    /// <summary>
    /// Output memory allocation ID.
    /// </summary>
    public string OutputAllocationId { get; init; }

    /// <summary>
    /// Optional kernel execution parameters (work group size, constants, etc.).
    /// </summary>
    public Dictionary<string, object>? Parameters { get; init; }

    public ComputeMessage(string kernelId, string inputAllocationId, string outputAllocationId, Dictionary<string, object>? parameters = null)
    {
        KernelId = kernelId;
        InputAllocationId = inputAllocationId;
        OutputAllocationId = outputAllocationId;
        Parameters = parameters;
    }
}

/// <summary>
/// Response message for successful kernel execution.
/// </summary>
public sealed record ComputeResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Whether execution was successful.
    /// </summary>
    public bool Success { get; init; }

    /// <summary>
    /// Kernel execution time in microseconds.
    /// </summary>
    public double KernelTimeMicroseconds { get; init; }

    /// <summary>
    /// Total execution time (including memory transfers) in microseconds.
    /// </summary>
    public double TotalTimeMicroseconds { get; init; }

    /// <summary>
    /// Error message if execution failed.
    /// </summary>
    public string? Error { get; init; }

    /// <summary>
    /// Whether kernel was cached (true) or newly compiled (false).
    /// </summary>
    public bool IsCacheHit { get; init; }

    public ComputeResponse(Guid originalRequestId, bool success, double kernelTimeMicroseconds, double totalTimeMicroseconds, string? error = null, bool isCacheHit = false)
    {
        OriginalRequestId = originalRequestId;
        Success = success;
        KernelTimeMicroseconds = kernelTimeMicroseconds;
        TotalTimeMicroseconds = totalTimeMicroseconds;
        Error = error;
        IsCacheHit = isCacheHit;
    }
}
