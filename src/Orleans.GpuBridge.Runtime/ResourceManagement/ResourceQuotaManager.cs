// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

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

/// <summary>
/// Manages resource quotas and allocation for multi-tenant scenarios
/// </summary>
public sealed class ResourceQuotaManager : IDisposable
{
    private readonly ILogger<ResourceQuotaManager> _logger;
    private readonly ResourceQuotaOptions _options;
    private readonly ConcurrentDictionary<string, TenantResourceUsage> _usageTracking;
    private readonly ConcurrentDictionary<string, TenantQuota> _quotas;
    private readonly SemaphoreSlim _allocationLock;
    private readonly Timer _resetTimer;
    private bool _disposed;
    
    public ResourceQuotaManager(
        ILogger<ResourceQuotaManager> logger,
        IOptions<ResourceQuotaOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        _usageTracking = new ConcurrentDictionary<string, TenantResourceUsage>();
        _quotas = new ConcurrentDictionary<string, TenantQuota>(_options.TenantQuotas);
        _allocationLock = new SemaphoreSlim(1, 1);
        
        // Start periodic quota reset timer
        _resetTimer = new Timer(
            ResetQuotas,
            null,
            _options.QuotaResetInterval,
            _options.QuotaResetInterval);
        
        _logger.LogInformation(
            "Resource quota manager initialized with {Count} tenant quotas",
            _quotas.Count);
    }
    
    /// <summary>
    /// Requests resource allocation for a tenant
    /// </summary>
    public async Task<ResourceAllocation?> RequestAllocationAsync(
        string tenantId,
        ResourceRequest request,
        CancellationToken ct = default)
    {
        if (!_options.EnableQuotas)
        {
            return new ResourceAllocation
            {
                TenantId = tenantId,
                Approved = true,
                AllocatedMemoryBytes = request.RequestedMemoryBytes,
                AllocatedKernels = request.RequestedKernels,
                AllocationId = Guid.NewGuid().ToString(),
                ExpiresAt = DateTime.UtcNow.AddHours(1)
            };
        }
        
        await _allocationLock.WaitAsync(ct);
        try
        {
            var quota = GetQuotaForTenant(tenantId);
            var usage = GetOrCreateUsage(tenantId);
            
            // Check if request exceeds quota
            if (!CanAllocate(quota, usage, request))
            {
                _logger.LogWarning(
                    "Resource allocation denied for tenant {TenantId}: quota exceeded",
                    tenantId);
                
                if (_options.EnforceHardLimits)
                {
                    return null;
                }
                
                // Soft limit - allow but log warning
                return CreateOverQuotaAllocation(tenantId, request);
            }
            
            // Update usage tracking
            usage.CurrentMemoryBytes += request.RequestedMemoryBytes;
            usage.ActiveKernels += request.RequestedKernels;
            usage.LastAllocationTime = DateTime.UtcNow;
            
            var allocation = new ResourceAllocation
            {
                TenantId = tenantId,
                Approved = true,
                AllocatedMemoryBytes = request.RequestedMemoryBytes,
                AllocatedKernels = request.RequestedKernels,
                AllocationId = Guid.NewGuid().ToString(),
                ExpiresAt = DateTime.UtcNow.Add(quota.MaxExecutionTime),
                Priority = quota.Priority
            };
            
            _logger.LogDebug(
                "Resource allocation approved for tenant {TenantId}: {Memory} bytes, {Kernels} kernels",
                tenantId, allocation.AllocatedMemoryBytes, allocation.AllocatedKernels);
            
            return allocation;
        }
        finally
        {
            _allocationLock.Release();
        }
    }
    
    /// <summary>
    /// Releases previously allocated resources
    /// </summary>
    public async Task ReleaseAllocationAsync(
        string tenantId,
        string allocationId,
        long memoryBytes,
        int kernels)
    {
        if (!_options.EnableQuotas) return;
        
        await _allocationLock.WaitAsync();
        try
        {
            if (_usageTracking.TryGetValue(tenantId, out var usage))
            {
                usage.CurrentMemoryBytes = Math.Max(0, usage.CurrentMemoryBytes - memoryBytes);
                usage.ActiveKernels = Math.Max(0, usage.ActiveKernels - kernels);
                
                _logger.LogDebug(
                    "Released resources for tenant {TenantId}: {Memory} bytes, {Kernels} kernels",
                    tenantId, memoryBytes, kernels);
            }
        }
        finally
        {
            _allocationLock.Release();
        }
    }
    
    /// <summary>
    /// Gets current usage statistics for a tenant
    /// </summary>
    public TenantResourceUsage GetUsage(string tenantId)
    {
        return _usageTracking.GetOrAdd(tenantId, _ => new TenantResourceUsage { TenantId = tenantId });
    }
    
    /// <summary>
    /// Gets all tenant usage statistics
    /// </summary>
    public IReadOnlyDictionary<string, TenantResourceUsage> GetAllUsage()
    {
        return _usageTracking;
    }
    
    /// <summary>
    /// Updates quota for a tenant
    /// </summary>
    public void UpdateQuota(string tenantId, TenantQuota quota)
    {
        _quotas[tenantId] = quota;
        _logger.LogInformation(
            "Updated quota for tenant {TenantId}: {Memory} bytes, {Kernels} kernels",
            tenantId, quota.MaxMemoryBytes, quota.MaxConcurrentKernels);
    }
    
    private TenantQuota GetQuotaForTenant(string tenantId)
    {
        return _quotas.GetValueOrDefault(tenantId) ?? _options.DefaultQuota;
    }
    
    private TenantResourceUsage GetOrCreateUsage(string tenantId)
    {
        return _usageTracking.GetOrAdd(tenantId, id => new TenantResourceUsage { TenantId = id });
    }
    
    private bool CanAllocate(TenantQuota quota, TenantResourceUsage usage, ResourceRequest request)
    {
        // Check memory limit
        if (usage.CurrentMemoryBytes + request.RequestedMemoryBytes > quota.MaxMemoryBytes)
            return false;
        
        // Check concurrent kernel limit
        if (usage.ActiveKernels + request.RequestedKernels > quota.MaxConcurrentKernels)
            return false;
        
        // Check batch size limit
        if (request.BatchSize > quota.MaxBatchSize)
            return false;
        
        return true;
    }
    
    private ResourceAllocation CreateOverQuotaAllocation(string tenantId, ResourceRequest request)
    {
        return new ResourceAllocation
        {
            TenantId = tenantId,
            Approved = true,
            AllocatedMemoryBytes = request.RequestedMemoryBytes,
            AllocatedKernels = request.RequestedKernels,
            AllocationId = Guid.NewGuid().ToString(),
            ExpiresAt = DateTime.UtcNow.AddMinutes(1), // Shorter expiry for over-quota
            Priority = -1, // Lower priority
            IsOverQuota = true
        };
    }
    
    private void ResetQuotas(object? state)
    {
        if (_disposed) return;
        
        _logger.LogInformation("Resetting tenant resource usage counters");
        
        foreach (var usage in _usageTracking.Values)
        {
            usage.TotalMemoryBytesUsed += usage.CurrentMemoryBytes;
            usage.TotalKernelsExecuted += usage.ActiveKernels;
            usage.ResetCount++;
            usage.LastResetTime = DateTime.UtcNow;
        }
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _resetTimer?.Dispose();
        _allocationLock?.Dispose();
    }
}

/// <summary>
/// Resource request from a tenant
/// </summary>
public sealed class ResourceRequest
{
    public long RequestedMemoryBytes { get; set; }
    public int RequestedKernels { get; set; }
    public int BatchSize { get; set; }
    public TimeSpan EstimatedDuration { get; set; }
}

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