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
    /// <summary>
    /// Gets or sets the unique identifier for the tenant.
    /// </summary>
    public string TenantId { get; set; } = default!;

    /// <summary>
    /// Gets or sets the maximum memory in bytes that the tenant can allocate.
    /// </summary>
    public long MaxMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of GPU kernels that can execute concurrently for this tenant.
    /// </summary>
    public int MaxConcurrentKernels { get; set; }

    /// <summary>
    /// Gets or sets the maximum batch size for kernel execution operations.
    /// </summary>
    public int MaxBatchSize { get; set; }

    /// <summary>
    /// Gets or sets the percentage of CPU resources allocated to this tenant.
    /// </summary>
    public double CpuSharePercentage { get; set; }

    /// <summary>
    /// Gets or sets the percentage of GPU resources allocated to this tenant.
    /// </summary>
    public double GpuSharePercentage { get; set; }

    /// <summary>
    /// Gets or sets the maximum execution time allowed for kernel operations.
    /// </summary>
    public TimeSpan MaxExecutionTime { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Gets or sets the priority level for this tenant. Higher values indicate higher priority.
    /// </summary>
    public int Priority { get; set; } = 0;
}

/// <summary>
/// Options for resource quota management
/// </summary>
public sealed class ResourceQuotaOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether resource quotas are enabled.
    /// </summary>
    public bool EnableQuotas { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether hard limits are enforced. When true, operations exceeding quotas will be rejected.
    /// </summary>
    public bool EnforceHardLimits { get; set; } = false;

    /// <summary>
    /// Gets or sets the interval at which quota usage counters are reset.
    /// </summary>
    public TimeSpan QuotaResetInterval { get; set; } = TimeSpan.FromHours(1);

    /// <summary>
    /// Gets the dictionary of tenant-specific quota configurations, keyed by tenant identifier.
    /// </summary>
    public Dictionary<string, TenantQuota> TenantQuotas { get; } = new();

    /// <summary>
    /// Gets or sets the default quota configuration applied to tenants without explicit quota definitions.
    /// </summary>
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
