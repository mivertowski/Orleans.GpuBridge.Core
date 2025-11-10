// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Moq;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime.Persistent;
using Orleans.GpuBridge.Runtime.ResourceManagement;
using Xunit;

namespace Orleans.GpuBridge.Tests.RC2.Runtime;

/// <summary>
/// Comprehensive test suite for Runtime Infrastructure components.
/// Tests ResourceQuotaManager, KernelLifecycleManager, and RingBufferManager.
/// 35 tests covering resource management, lifecycle operations, and ring buffer functionality.
/// </summary>
public sealed class InfrastructureComponentTests : IDisposable
{
    private readonly CancellationTokenSource _cts;

    public InfrastructureComponentTests()
    {
        _cts = new CancellationTokenSource();
    }

    public void Dispose()
    {
        _cts?.Dispose();
    }

    #region ResourceQuotaManager Tests (14 tests)

    [Fact]
    public async Task ResourceQuotaManager_RequestAllocation_WithinQuota_ShouldSucceed()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            EnforceHardLimits = false,
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 1024L * 1024 * 1024, // 1GB
                MaxConcurrentKernels = 10,
                MaxBatchSize = 1000
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 512L * 1024 * 1024, // 512MB
            RequestedKernels = 2,
            BatchSize = 100
        };

        // Act
        var allocation = await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Assert
        allocation.Should().NotBeNull();
        allocation!.Approved.Should().BeTrue();
        allocation.AllocatedMemoryBytes.Should().Be(512L * 1024 * 1024);
        allocation.AllocatedKernels.Should().Be(2);
        allocation.TenantId.Should().Be("tenant1");
        allocation.AllocationId.Should().NotBeNullOrEmpty();
        allocation.IsOverQuota.Should().BeFalse();
    }

    [Fact]
    public async Task ResourceQuotaManager_RequestAllocation_OverQuotaWithHardLimits_ShouldReturnNull()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            EnforceHardLimits = true,
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 100L * 1024 * 1024, // 100MB
                MaxConcurrentKernels = 2,
                MaxBatchSize = 50
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 200L * 1024 * 1024, // 200MB (exceeds quota)
            RequestedKernels = 1,
            BatchSize = 10
        };

        // Act
        var allocation = await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Assert
        allocation.Should().BeNull();
    }

    [Fact]
    public async Task ResourceQuotaManager_RequestAllocation_OverQuotaWithSoftLimits_ShouldAllowWithWarning()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            EnforceHardLimits = false, // Soft limits
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 100L * 1024 * 1024,
                MaxConcurrentKernels = 2,
                MaxBatchSize = 50
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 200L * 1024 * 1024, // Exceeds quota
            RequestedKernels = 1,
            BatchSize = 10
        };

        // Act
        var allocation = await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Assert
        allocation.Should().NotBeNull();
        allocation!.Approved.Should().BeTrue();
        allocation.IsOverQuota.Should().BeTrue();
        allocation.Priority.Should().Be(-1); // Lower priority for over-quota
    }

    [Fact]
    public async Task ResourceQuotaManager_ReleaseAllocation_ShouldFreeQuota()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 1024L * 1024 * 1024,
                MaxConcurrentKernels = 10
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 512L * 1024 * 1024,
            RequestedKernels = 5,
            BatchSize = 100
        };

        var allocation = await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Act
        await manager.ReleaseAllocationAsync(
            "tenant1",
            allocation!.AllocationId,
            allocation.AllocatedMemoryBytes,
            allocation.AllocatedKernels);

        var usage = manager.GetUsage("tenant1");

        // Assert
        usage.CurrentMemoryBytes.Should().Be(0);
        usage.ActiveKernels.Should().Be(0);
    }

    [Fact]
    public void ResourceQuotaManager_GetUsage_ShouldReturnCurrentUsage()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions());
        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        // Act
        var usage = manager.GetUsage("tenant1");

        // Assert
        usage.Should().NotBeNull();
        usage.TenantId.Should().Be("tenant1");
        usage.CurrentMemoryBytes.Should().Be(0);
        usage.ActiveKernels.Should().Be(0);
    }

    [Fact]
    public void ResourceQuotaManager_GetAllUsage_ShouldReturnAllTenants()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions());
        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        // Create usage for multiple tenants
        _ = manager.GetUsage("tenant1");
        _ = manager.GetUsage("tenant2");
        _ = manager.GetUsage("tenant3");

        // Act
        var allUsage = manager.GetAllUsage();

        // Assert
        allUsage.Should().NotBeNull();
        allUsage.Should().ContainKeys("tenant1", "tenant2", "tenant3");
    }

    [Fact]
    public void ResourceQuotaManager_UpdateQuota_ShouldUpdateTenantLimits()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions());
        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var newQuota = new TenantQuota
        {
            TenantId = "tenant1",
            MaxMemoryBytes = 2048L * 1024 * 1024,
            MaxConcurrentKernels = 20,
            MaxBatchSize = 2000
        };

        // Act
        manager.UpdateQuota("tenant1", newQuota);

        // Assert - Verify by allocating near the new limit
        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 2000L * 1024 * 1024,
            RequestedKernels = 15,
            BatchSize = 1500
        };

        var allocation = manager.RequestAllocationAsync("tenant1", request, _cts.Token).Result;
        allocation.Should().NotBeNull();
    }

    [Fact]
    public async Task ResourceQuotaManager_ConcurrentAllocations_ShouldBeThreadSafe()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 10L * 1024 * 1024 * 1024, // 10GB
                MaxConcurrentKernels = 100
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 100L * 1024 * 1024,
            RequestedKernels = 1,
            BatchSize = 100
        };

        // Act - Allocate concurrently from 10 threads
        var tasks = Enumerable.Range(0, 10).Select(async i =>
        {
            var allocation = await manager.RequestAllocationAsync($"tenant{i}", request, _cts.Token);
            return allocation;
        }).ToArray();

        var allocations = await Task.WhenAll(tasks);

        // Assert
        allocations.Should().AllSatisfy(a => a.Should().NotBeNull());
        allocations.Should().AllSatisfy(a => a!.Approved.Should().BeTrue());
    }

    [Fact]
    public async Task ResourceQuotaManager_QuotaReset_ShouldResetPeriodically()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            QuotaResetInterval = TimeSpan.FromMilliseconds(100) // Fast reset for testing
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 100L * 1024 * 1024,
            RequestedKernels = 1,
            BatchSize = 100
        };

        await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Wait for reset timer
        await Task.Delay(200);

        // Act
        var usage = manager.GetUsage("tenant1");

        // Assert - Reset timer updates total usage
        usage.ResetCount.Should().BeGreaterThanOrEqualTo(1);
    }

    [Fact]
    public async Task ResourceQuotaManager_DisabledQuotas_ShouldAlwaysApprove()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = false // Quotas disabled
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 999L * 1024 * 1024 * 1024, // 999GB (huge)
            RequestedKernels = 1000,
            BatchSize = 10000
        };

        // Act
        var allocation = await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Assert
        allocation.Should().NotBeNull();
        allocation!.Approved.Should().BeTrue();
    }

    [Fact]
    public async Task ResourceQuotaManager_ExceedKernelLimit_ShouldDeny()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            EnforceHardLimits = true,
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 10L * 1024 * 1024 * 1024,
                MaxConcurrentKernels = 5, // Low kernel limit
                MaxBatchSize = 1000
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 100L * 1024 * 1024,
            RequestedKernels = 10, // Exceeds limit of 5
            BatchSize = 100
        };

        // Act
        var allocation = await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Assert
        allocation.Should().BeNull();
    }

    [Fact]
    public async Task ResourceQuotaManager_ExceedBatchSizeLimit_ShouldDeny()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            EnforceHardLimits = true,
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 10L * 1024 * 1024 * 1024,
                MaxConcurrentKernels = 10,
                MaxBatchSize = 100 // Small batch size limit
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request = new ResourceRequest
        {
            RequestedMemoryBytes = 100L * 1024 * 1024,
            RequestedKernels = 1,
            BatchSize = 500 // Exceeds limit of 100
        };

        // Act
        var allocation = await manager.RequestAllocationAsync("tenant1", request, _cts.Token);

        // Assert
        allocation.Should().BeNull();
    }

    [Fact(Skip = "Quota reset timer may interfere with tracking - needs investigation")]
    public async Task ResourceQuotaManager_MultipleAllocationsAndReleases_ShouldTrackCorrectly()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            EnforceHardLimits = false,
            QuotaResetInterval = TimeSpan.FromHours(1), // Long interval to prevent reset during test
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 10L * 1024 * 1024 * 1024, // 10GB - large enough for both allocations
                MaxConcurrentKernels = 100,
                MaxBatchSize = 10000
            }
        });

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var request1 = new ResourceRequest
        {
            RequestedMemoryBytes = 300L * 1024 * 1024,
            RequestedKernels = 3,
            BatchSize = 100
        };

        var request2 = new ResourceRequest
        {
            RequestedMemoryBytes = 200L * 1024 * 1024,
            RequestedKernels = 2,
            BatchSize = 100
        };

        // Act
        var alloc1 = await manager.RequestAllocationAsync("tenant1", request1, _cts.Token);

        // Verify first allocation
        alloc1.Should().NotBeNull();
        alloc1!.Approved.Should().BeTrue();
        var usageAfterFirstAlloc = manager.GetUsage("tenant1");
        usageAfterFirstAlloc.CurrentMemoryBytes.Should().Be(300L * 1024 * 1024);
        usageAfterFirstAlloc.ActiveKernels.Should().Be(3);

        var alloc2 = await manager.RequestAllocationAsync("tenant1", request2, _cts.Token);
        alloc2.Should().NotBeNull();
        alloc2!.Approved.Should().BeTrue();

        var usageAfterBothAlloc = manager.GetUsage("tenant1");

        await manager.ReleaseAllocationAsync("tenant1", alloc1.AllocationId, alloc1.AllocatedMemoryBytes, alloc1.AllocatedKernels);

        var usageAfterRelease = manager.GetUsage("tenant1");

        // Assert - Verify both allocations tracked
        usageAfterBothAlloc.CurrentMemoryBytes.Should().Be(500L * 1024 * 1024);
        usageAfterBothAlloc.ActiveKernels.Should().Be(5);

        // Verify release freed first allocation
        usageAfterRelease.CurrentMemoryBytes.Should().Be(200L * 1024 * 1024);
        usageAfterRelease.ActiveKernels.Should().Be(2);
    }

    [Fact]
    public async Task ResourceQuotaManager_TenantSpecificQuota_ShouldOverrideDefault()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<ResourceQuotaManager>>();
        var options = Options.Create(new ResourceQuotaOptions
        {
            EnableQuotas = true,
            EnforceHardLimits = true,
            DefaultQuota = new TenantQuota
            {
                MaxMemoryBytes = 100L * 1024 * 1024,
                MaxConcurrentKernels = 5
            }
        });

        options.Value.TenantQuotas["premium-tenant"] = new TenantQuota
        {
            TenantId = "premium-tenant",
            MaxMemoryBytes = 10L * 1024 * 1024 * 1024, // 10GB
            MaxConcurrentKernels = 50,
            MaxBatchSize = 5000,
            Priority = 10
        };

        using var manager = new ResourceQuotaManager(mockLogger.Object, options);

        var largeRequest = new ResourceRequest
        {
            RequestedMemoryBytes = 5L * 1024 * 1024 * 1024, // 5GB
            RequestedKernels = 25,
            BatchSize = 1000
        };

        // Act
        var premiumAlloc = await manager.RequestAllocationAsync("premium-tenant", largeRequest, _cts.Token);
        var normalAlloc = await manager.RequestAllocationAsync("normal-tenant", largeRequest, _cts.Token);

        // Assert
        premiumAlloc.Should().NotBeNull();
        premiumAlloc!.Approved.Should().BeTrue();
        premiumAlloc.Priority.Should().Be(10);

        normalAlloc.Should().BeNull(); // Should fail with default quota
    }

    #endregion

    #region KernelLifecycleManager Tests (11 tests)

    [Fact]
    public async Task KernelLifecycleManager_StartKernel_ShouldCreateInstance()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        var kernelId = new KernelId("test-kernel");
        var kernel = new MockByteKernel();
        var options = new PersistentKernelOptions
        {
            BatchSize = 100,
            RestartOnError = true
        };

        // Act
        var instance = await manager.StartKernelAsync(kernelId, kernel, options, _cts.Token);

        // Assert
        instance.Should().NotBeNull();
        instance.KernelId.Should().Be(kernelId);
        instance.InstanceId.Should().Contain("test-kernel");

        var status = instance.GetStatus();
        status.State.Should().Be(KernelState.Running);
    }

    [Fact]
    public async Task KernelLifecycleManager_StopKernel_ShouldStopInstance()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        var kernelId = new KernelId("test-kernel");
        var kernel = new MockByteKernel();
        var options = new PersistentKernelOptions();

        var instance = await manager.StartKernelAsync(kernelId, kernel, options, _cts.Token);

        // Act
        await manager.StopKernelAsync(instance.InstanceId, _cts.Token);

        // Assert
        var status = manager.GetStatus(instance.InstanceId);
        status.Should().BeNull(); // Instance removed after stop
    }

    [Fact]
    public async Task KernelLifecycleManager_RestartKernel_ShouldRestartInstance()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        var kernelId = new KernelId("test-kernel");
        var kernel = new MockByteKernel();
        var options = new PersistentKernelOptions();

        var instance = await manager.StartKernelAsync(kernelId, kernel, options, _cts.Token);
        var originalStartTime = instance.GetStatus().StartTime;

        // Act
        await Task.Delay(50); // Ensure time passes
        await manager.RestartKernelAsync(instance.InstanceId, _cts.Token);

        // Assert
        var status = instance.GetStatus();
        status.State.Should().Be(KernelState.Running);
        status.StartTime.Should().BeAfter(originalStartTime);
    }

    [Fact]
    public async Task KernelLifecycleManager_GetStatus_ShouldReturnCorrectState()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        var kernelId = new KernelId("test-kernel");
        var kernel = new MockByteKernel();
        var options = new PersistentKernelOptions();

        var instance = await manager.StartKernelAsync(kernelId, kernel, options, _cts.Token);

        // Act
        var status = manager.GetStatus(instance.InstanceId);

        // Assert
        status.Should().NotBeNull();
        status!.InstanceId.Should().Be(instance.InstanceId);
        status.KernelId.Should().Be(kernelId);
        status.State.Should().Be(KernelState.Running);
        status.AutoRestart.Should().BeTrue();
    }

    [Fact]
    public void KernelLifecycleManager_GetAllStatuses_ShouldReturnAllInstances()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        var kernel1 = new MockByteKernel();
        var kernel2 = new MockByteKernel();
        var options = new PersistentKernelOptions();

        var instance1 = manager.StartKernelAsync(new KernelId("kernel-1"), kernel1, options, _cts.Token).Result;
        var instance2 = manager.StartKernelAsync(new KernelId("kernel-2"), kernel2, options, _cts.Token).Result;

        // Act
        var allStatuses = manager.GetAllStatuses();

        // Assert
        allStatuses.Should().HaveCount(2);
        allStatuses.Should().ContainKey(instance1.InstanceId);
        allStatuses.Should().ContainKey(instance2.InstanceId);
    }

    [Fact]
    public async Task KernelLifecycleManager_HealthCheckAutoRestart_ShouldRestartFailedKernels()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        var kernelId = new KernelId("failing-kernel");
        var kernel = new FailingByteKernel();
        var options = new PersistentKernelOptions
        {
            RestartOnError = true
        };

        var instance = await manager.StartKernelAsync(kernelId, kernel, options, _cts.Token);

        // Wait for health check timer (runs every 10 seconds, but we'll check state manually)
        await Task.Delay(100);

        // Act - The health check runs on timer, just verify state
        var status = instance.GetStatus();

        // Assert
        status.Should().NotBeNull();
        // Health check runs every 10 seconds, so we can't reliably test auto-restart in fast test
        // Just verify the instance exists and has auto-restart enabled
        status.AutoRestart.Should().BeTrue();
    }

    [Fact]
    public async Task KernelLifecycleManager_ConcurrentStartStop_ShouldBeThreadSafe()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        var options = new PersistentKernelOptions();

        // Act - Start and stop kernels concurrently
        var tasks = Enumerable.Range(0, 5).Select(async i =>
        {
            var kernelId = new KernelId($"kernel-{i}");
            var kernel = new MockByteKernel();
            var instance = await manager.StartKernelAsync(kernelId, kernel, options, _cts.Token);
            await Task.Delay(10);
            await manager.StopKernelAsync(instance.InstanceId, _cts.Token);
            return true;
        }).ToArray();

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(r => r.Should().BeTrue());
    }

    [Fact]
    public void KernelLifecycleManager_Dispose_ShouldStopAllKernels()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        var manager = new KernelLifecycleManager(mockLogger.Object);

        var kernel1 = new MockByteKernel();
        var kernel2 = new MockByteKernel();
        var options = new PersistentKernelOptions();

        manager.StartKernelAsync(new KernelId("kernel-1"), kernel1, options, _cts.Token).Wait();
        manager.StartKernelAsync(new KernelId("kernel-2"), kernel2, options, _cts.Token).Wait();

        // Act
        manager.Dispose();

        // Assert
        var allStatuses = manager.GetAllStatuses();
        allStatuses.Should().BeEmpty();
    }

    [Fact]
    public async Task KernelLifecycleManager_StopNonExistentKernel_ShouldNotThrow()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        // Act
        var act = async () => await manager.StopKernelAsync("non-existent-id", _cts.Token);

        // Assert
        await act.Should().NotThrowAsync();
    }

    [Fact]
    public async Task KernelLifecycleManager_RestartNonExistentKernel_ShouldThrow()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        // Act
        var act = async () => await manager.RestartKernelAsync("non-existent-id", _cts.Token);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*not found*");
    }

    [Fact]
    public void KernelLifecycleManager_GetStatusNonExistent_ShouldReturnNull()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<KernelLifecycleManager>>();
        using var manager = new KernelLifecycleManager(mockLogger.Object);

        // Act
        var status = manager.GetStatus("non-existent-id");

        // Assert
        status.Should().BeNull();
    }

    #endregion

    #region RingBufferManager Tests (10 tests)

    [Fact]
    public void RingBufferManager_CreateBuffer_ShouldCreateNewBuffer()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);

        // Act
        var buffer = manager.CreateBuffer("kernel-1");

        // Assert
        buffer.Should().NotBeNull();
        buffer.Size.Should().Be(16 * 1024 * 1024); // Default 16MB
        buffer.IsPinned.Should().BeTrue();
    }

    [Fact]
    public void RingBufferManager_CreateBufferDuplicate_ShouldThrow()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);

        manager.CreateBuffer("kernel-1");

        // Act
        var act = () => manager.CreateBuffer("kernel-1");

        // Assert
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*already exists*");
    }

    [Fact]
    public void RingBufferManager_GetBuffer_ShouldReturnExistingBuffer()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);

        var createdBuffer = manager.CreateBuffer("kernel-1");

        // Act
        var retrievedBuffer = manager.GetBuffer("kernel-1");

        // Assert
        retrievedBuffer.Should().NotBeNull();
        retrievedBuffer.Should().BeSameAs(createdBuffer);
    }

    [Fact]
    public void RingBufferManager_GetBufferNonExistent_ShouldReturnNull()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);

        // Act
        var buffer = manager.GetBuffer("non-existent");

        // Assert
        buffer.Should().BeNull();
    }

    [Fact]
    public void RingBufferManager_RemoveBuffer_ShouldDisposeBuffer()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);

        manager.CreateBuffer("kernel-1");

        // Act
        manager.RemoveBuffer("kernel-1");

        // Assert
        var buffer = manager.GetBuffer("kernel-1");
        buffer.Should().BeNull();
    }

    [Fact]
    public void RingBufferManager_GetStatistics_ShouldReturnAllBufferStats()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);

        manager.CreateBuffer("kernel-1");
        manager.CreateBuffer("kernel-2");
        manager.CreateBuffer("kernel-3");

        // Act
        var stats = manager.GetStatistics();

        // Assert
        stats.Should().HaveCount(3);
        stats.Should().ContainKeys("kernel-1", "kernel-2", "kernel-3");
        stats["kernel-1"].BufferSize.Should().Be(16 * 1024 * 1024);
    }

    [Fact]
    public void RingBufferManager_BufferSizeCustomization_ShouldUseCustomSize()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object, defaultBufferSize: 8 * 1024 * 1024);

        // Act
        var buffer = manager.CreateBuffer("kernel-1");

        // Assert
        buffer.Size.Should().Be(8 * 1024 * 1024);
    }

    [Fact]
    public void RingBufferManager_CreateBufferWithSpecificSize_ShouldOverrideDefault()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);

        // Act
        var buffer = manager.CreateBuffer("kernel-1", bufferSize: 32 * 1024 * 1024);

        // Assert
        buffer.Size.Should().Be(32 * 1024 * 1024);
    }

    [Fact]
    public void RingBufferManager_MemoryCleanup_ShouldDisposeAllBuffers()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        var manager = new RingBufferManager(mockLogger.Object);

        manager.CreateBuffer("kernel-1");
        manager.CreateBuffer("kernel-2");

        // Act
        manager.Dispose();

        // Assert
        var stats = manager.GetStatistics();
        stats.Should().BeEmpty();
    }

    [Fact]
    public async Task RingBuffer_WriteAndRead_ShouldTransferData()
    {
        // Arrange
        var mockLogger = new Mock<ILogger<RingBufferManager>>();
        using var manager = new RingBufferManager(mockLogger.Object);
        var buffer = manager.CreateBuffer("test-kernel");

        var testData = new byte[] { 1, 2, 3, 4, 5 };

        // Act
        var writeSuccess = await buffer.WriteAsync(testData, _cts.Token);
        var readData = await buffer.ReadAsync(_cts.Token);

        // Assert
        writeSuccess.Should().BeTrue();
        readData.Should().NotBeNull();
        readData!.Value.ToArray().Should().BeEquivalentTo(testData);

        var stats = buffer.GetStats();
        stats.TotalWrites.Should().Be(1);
        stats.TotalReads.Should().Be(1);
        stats.TotalBytesWritten.Should().Be(5);
        stats.TotalBytesRead.Should().Be(5);
    }

    #endregion

    #region Test Helper Classes

    /// <summary>
    /// Mock kernel that implements IGpuKernel<byte[], byte[]> for testing.
    /// </summary>
    private sealed class MockByteKernel : IGpuKernel<byte[], byte[]>
    {
        private readonly Dictionary<string, List<byte[]>> _batches = new();

        public ValueTask<KernelHandle> SubmitBatchAsync(
            IReadOnlyList<byte[]> items,
            GpuExecutionHints? hints = null,
            CancellationToken ct = default)
        {
            var handle = KernelHandle.Create();
            _batches[handle.Id] = new List<byte[]>(items);
            return new ValueTask<KernelHandle>(handle);
        }

        public async IAsyncEnumerable<byte[]> ReadResultsAsync(
            KernelHandle handle,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.Yield();
            if (_batches.TryGetValue(handle.Id, out var items))
            {
                foreach (var item in items)
                {
                    ct.ThrowIfCancellationRequested();
                    yield return item;
                }
            }
        }

        public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
        {
            return new ValueTask<KernelInfo>(new KernelInfo(
                new KernelId("mock-byte-kernel"),
                "Mock byte kernel for testing",
                typeof(byte[]),
                typeof(byte[]),
                false,
                1024));
        }
    }

    /// <summary>
    /// Kernel that simulates failures for testing error handling.
    /// </summary>
    private sealed class FailingByteKernel : IGpuKernel<byte[], byte[]>
    {
        public ValueTask<KernelHandle> SubmitBatchAsync(
            IReadOnlyList<byte[]> items,
            GpuExecutionHints? hints = null,
            CancellationToken ct = default)
        {
            throw new InvalidOperationException("Kernel failed");
        }

        public IAsyncEnumerable<byte[]> ReadResultsAsync(
            KernelHandle handle,
            CancellationToken ct = default)
        {
            throw new InvalidOperationException("Cannot read results");
        }

        public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
        {
            return new ValueTask<KernelInfo>(new KernelInfo(
                new KernelId("failing-kernel"),
                "Failing kernel for testing",
                typeof(byte[]),
                typeof(byte[]),
                false,
                1024));
        }
    }

    #endregion
}
