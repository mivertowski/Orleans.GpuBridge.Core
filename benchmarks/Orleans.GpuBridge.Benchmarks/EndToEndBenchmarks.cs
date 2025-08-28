// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.ResourceManagement;

namespace Orleans.GpuBridge.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class EndToEndBenchmarks
{
    private IServiceProvider _serviceProvider = null!;
    private ResourceQuotaManager _quotaManager = null!;
    private MemoryPoolManager _memoryManager = null!;
    
    [GlobalSetup]
    public void Setup()
    {
        var services = new ServiceCollection();
        
        services.AddSingleton(NullLoggerFactory.Instance);
        services.AddLogging();
        services.AddGpuBridge(options =>
        {
            options.PreferGpu = false; // Use CPU for benchmarks
            options.EnableFallback = true;
        });
        
        _serviceProvider = services.BuildServiceProvider();
        
        _quotaManager = new ResourceQuotaManager(
            NullLogger<ResourceQuotaManager>.Instance,
            Microsoft.Extensions.Options.Options.Create(new ResourceQuotaOptions()));
        
        _memoryManager = new MemoryPoolManager(NullLoggerFactory.Instance);
    }
    
    [GlobalCleanup]
    public void Cleanup()
    {
        _quotaManager?.Dispose();
        _memoryManager?.Dispose();
        (_serviceProvider as IDisposable)?.Dispose();
    }
    
    [Benchmark]
    public async Task ResourceAllocation_SingleTenant()
    {
        var allocation = await _quotaManager.RequestAllocationAsync(
            "tenant1",
            new ResourceRequest
            {
                RequestedMemoryBytes = 1024 * 1024,
                RequestedKernels = 1,
                BatchSize = 100,
                EstimatedDuration = TimeSpan.FromSeconds(1)
            });
        
        if (allocation != null)
        {
            await _quotaManager.ReleaseAllocationAsync(
                "tenant1",
                allocation.AllocationId,
                allocation.AllocatedMemoryBytes,
                allocation.AllocatedKernels);
        }
    }
    
    [Benchmark]
    public async Task ResourceAllocation_MultiTenant()
    {
        var tasks = new List<Task>();
        
        for (int i = 0; i < 10; i++)
        {
            var tenantId = $"tenant{i}";
            tasks.Add(Task.Run(async () =>
            {
                var allocation = await _quotaManager.RequestAllocationAsync(
                    tenantId,
                    new ResourceRequest
                    {
                        RequestedMemoryBytes = 100 * 1024,
                        RequestedKernels = 1,
                        BatchSize = 10,
                        EstimatedDuration = TimeSpan.FromMilliseconds(100)
                    });
                
                if (allocation != null)
                {
                    await Task.Delay(10);
                    await _quotaManager.ReleaseAllocationAsync(
                        tenantId,
                        allocation.AllocationId,
                        allocation.AllocatedMemoryBytes,
                        allocation.AllocatedKernels);
                }
            }));
        }
        
        await Task.WhenAll(tasks);
    }
    
    [Benchmark]
    public void MemoryPool_MultiType()
    {
        var floatPool = _memoryManager.GetPool<float>();
        var doublePool = _memoryManager.GetPool<double>();
        var intPool = _memoryManager.GetPool<int>();
        
        var floatMem = floatPool.Rent(1024);
        var doubleMem = doublePool.Rent(512);
        var intMem = intPool.Rent(2048);
        
        floatMem.Dispose();
        doubleMem.Dispose();
        intMem.Dispose();
    }
    
    [Benchmark]
    public MemoryPoolHealth GetMemoryHealth()
    {
        // Allocate some memory first
        var pool = _memoryManager.GetPool<float>();
        var memories = new List<Orleans.GpuBridge.Abstractions.Memory.IGpuMemory<float>>();
        
        for (int i = 0; i < 10; i++)
        {
            memories.Add(pool.Rent(1024));
        }
        
        var health = _memoryManager.GetHealthStatus();
        
        foreach (var mem in memories)
        {
            mem.Dispose();
        }
        
        return health;
    }
}