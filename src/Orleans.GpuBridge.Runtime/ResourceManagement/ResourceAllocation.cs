// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.ResourceManagement;

/// <summary>
/// Resource allocation result
/// </summary>
public sealed class ResourceAllocation
{
    public string TenantId { get; set; } = default!;
    public string AllocationId { get; set; } = default!;
    public bool Approved { get; set; }
    public long AllocatedMemoryBytes { get; set; }
    public int AllocatedKernels { get; set; }
    public DateTime ExpiresAt { get; set; }
    public int Priority { get; set; }
    public bool IsOverQuota { get; set; }
}
