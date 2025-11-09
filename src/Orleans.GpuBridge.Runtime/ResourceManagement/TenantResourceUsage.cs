// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.ResourceManagement;

/// <summary>
/// Tracks resource usage for a tenant
/// </summary>
public sealed class TenantResourceUsage
{
    public string TenantId { get; set; } = default!;
    public long CurrentMemoryBytes { get; set; }
    public long TotalMemoryBytesUsed { get; set; }
    public int ActiveKernels { get; set; }
    public int TotalKernelsExecuted { get; set; }
    public DateTime LastAllocationTime { get; set; }
    public DateTime LastResetTime { get; set; }
    public int ResetCount { get; set; }
}
