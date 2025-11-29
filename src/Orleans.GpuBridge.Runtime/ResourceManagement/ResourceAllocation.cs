// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.ResourceManagement;

/// <summary>
/// Resource allocation result
/// </summary>
public sealed class ResourceAllocation
{
    /// <summary>
    /// Gets or sets the unique identifier of the tenant requesting the allocation.
    /// </summary>
    public string TenantId { get; set; } = default!;

    /// <summary>
    /// Gets or sets the unique identifier for this specific resource allocation.
    /// </summary>
    public string AllocationId { get; set; } = default!;

    /// <summary>
    /// Gets or sets a value indicating whether the resource allocation request was approved.
    /// </summary>
    public bool Approved { get; set; }

    /// <summary>
    /// Gets or sets the amount of memory in bytes allocated to the tenant.
    /// </summary>
    public long AllocatedMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the number of kernel execution slots allocated to the tenant.
    /// </summary>
    public int AllocatedKernels { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when this allocation expires and resources should be reclaimed.
    /// </summary>
    public DateTime ExpiresAt { get; set; }

    /// <summary>
    /// Gets or sets the priority level of this allocation, with higher values indicating higher priority.
    /// </summary>
    public int Priority { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether this allocation exceeds the tenant's quota limits.
    /// </summary>
    public bool IsOverQuota { get; set; }
}
