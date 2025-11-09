// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime.Persistent;

/// <summary>
/// Status of a persistent kernel instance
/// </summary>
public sealed class KernelInstanceStatus
{
    public string InstanceId { get; init; } = string.Empty;
    public KernelId KernelId { get; init; } = new("unknown");
    public KernelState State { get; init; }
    public DateTime StartTime { get; init; }
    public DateTime LastActivity { get; init; }
    public long ProcessedBatches { get; init; }
    public long FailedBatches { get; init; }
    public long TotalBytesProcessed { get; init; }
    public string? LastError { get; init; }
    public bool AutoRestart { get; init; }
    public TimeSpan Uptime { get; init; }
    public double SuccessRate => ProcessedBatches > 0
        ? (ProcessedBatches - FailedBatches) / (double)ProcessedBatches * 100
        : 0;
}

/// <summary>
/// State of a kernel instance
/// </summary>
public enum KernelState
{
    Idle,
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed
}
