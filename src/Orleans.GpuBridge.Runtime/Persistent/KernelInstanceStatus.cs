// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime.Persistent;

/// <summary>
/// Status of a persistent kernel instance.
/// </summary>
public sealed class KernelInstanceStatus
{
    /// <summary>
    /// Gets the unique identifier of this kernel instance.
    /// </summary>
    public string InstanceId { get; init; } = string.Empty;

    /// <summary>
    /// Gets the kernel identifier.
    /// </summary>
    public KernelId KernelId { get; init; } = new("unknown");

    /// <summary>
    /// Gets the current state of the kernel instance.
    /// </summary>
    public KernelState State { get; init; }

    /// <summary>
    /// Gets the time when the kernel was started.
    /// </summary>
    public DateTime StartTime { get; init; }

    /// <summary>
    /// Gets the time of the last activity.
    /// </summary>
    public DateTime LastActivity { get; init; }

    /// <summary>
    /// Gets the number of successfully processed batches.
    /// </summary>
    public long ProcessedBatches { get; init; }

    /// <summary>
    /// Gets the number of failed batches.
    /// </summary>
    public long FailedBatches { get; init; }

    /// <summary>
    /// Gets the total number of bytes processed.
    /// </summary>
    public long TotalBytesProcessed { get; init; }

    /// <summary>
    /// Gets the last error message, if any.
    /// </summary>
    public string? LastError { get; init; }

    /// <summary>
    /// Gets whether auto-restart is enabled on error.
    /// </summary>
    public bool AutoRestart { get; init; }

    /// <summary>
    /// Gets the total uptime of the kernel instance.
    /// </summary>
    public TimeSpan Uptime { get; init; }

    /// <summary>
    /// Gets the success rate as a percentage (0-100).
    /// </summary>
    public double SuccessRate => ProcessedBatches > 0
        ? (ProcessedBatches - FailedBatches) / (double)ProcessedBatches * 100
        : 0;
}

/// <summary>
/// State of a kernel instance.
/// </summary>
public enum KernelState
{
    /// <summary>
    /// Kernel is idle and not processing.
    /// </summary>
    Idle,

    /// <summary>
    /// Kernel is starting up.
    /// </summary>
    Starting,

    /// <summary>
    /// Kernel is running and processing data.
    /// </summary>
    Running,

    /// <summary>
    /// Kernel is in the process of stopping.
    /// </summary>
    Stopping,

    /// <summary>
    /// Kernel has been stopped.
    /// </summary>
    Stopped,

    /// <summary>
    /// Kernel has encountered an error and failed.
    /// </summary>
    Failed
}
