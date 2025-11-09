// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Collections.Generic;

namespace Orleans.GpuBridge.Runtime.ResourceManagement;

/// <summary>
/// Resource quota configuration for a tenant
/// </summary>
public sealed class TenantQuota
{
    public string TenantId { get; set; } = default!;
    public long MaxMemoryBytes { get; set; }
    public int MaxConcurrentKernels { get; set; }
    public int MaxBatchSize { get; set; }
    public double CpuSharePercentage { get; set; }
    public double GpuSharePercentage { get; set; }
    public TimeSpan MaxExecutionTime { get; set; } = TimeSpan.FromMinutes(5);
    public int Priority { get; set; } = 0; // Higher values = higher priority
}

/// <summary>
/// Options for resource quota management
/// </summary>
public sealed class ResourceQuotaOptions
{
    public bool EnableQuotas { get; set; } = true;
    public bool EnforceHardLimits { get; set; } = false;
    public TimeSpan QuotaResetInterval { get; set; } = TimeSpan.FromHours(1);
    public Dictionary<string, TenantQuota> TenantQuotas { get; } = new();
    public TenantQuota DefaultQuota { get; set; } = new()
    {
        MaxMemoryBytes = 1024L * 1024 * 1024, // 1GB default
        MaxConcurrentKernels = 10,
        MaxBatchSize = 1000,
        CpuSharePercentage = 10.0,
        GpuSharePercentage = 10.0,
        MaxExecutionTime = TimeSpan.FromMinutes(5)
    };
}
