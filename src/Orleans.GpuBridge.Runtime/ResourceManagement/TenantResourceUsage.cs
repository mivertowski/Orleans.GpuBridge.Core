// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.ResourceManagement;

/// <summary>
/// Tracks resource usage for a tenant
/// </summary>
public sealed class TenantResourceUsage
{
    /// <summary>
    /// Gets or sets the unique identifier for the tenant.
    /// </summary>
    public string TenantId { get; set; } = default!;

    /// <summary>
    /// Gets or sets the current amount of memory in bytes actively allocated to the tenant.
    /// </summary>
    public long CurrentMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the cumulative total of memory in bytes used by the tenant since the last reset.
    /// </summary>
    public long TotalMemoryBytesUsed { get; set; }

    /// <summary>
    /// Gets or sets the number of kernels currently executing for the tenant.
    /// </summary>
    public int ActiveKernels { get; set; }

    /// <summary>
    /// Gets or sets the cumulative total number of kernels executed by the tenant since the last reset.
    /// </summary>
    public int TotalKernelsExecuted { get; set; }

    /// <summary>
    /// Gets or sets the timestamp of the most recent resource allocation for the tenant.
    /// </summary>
    public DateTime LastAllocationTime { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the tenant's resource usage counters were last reset.
    /// </summary>
    public DateTime LastResetTime { get; set; }

    /// <summary>
    /// Gets or sets the number of times the tenant's resource usage counters have been reset.
    /// </summary>
    public int ResetCount { get; set; }
}
