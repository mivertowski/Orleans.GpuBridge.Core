// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using NBomber.CSharp;
using NBomber.Contracts;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.ResourceManagement;

namespace Orleans.GpuBridge.Benchmarks;

/// <summary>
/// Load testing using NBomber
/// </summary>
public static class LoadTestRunner
{
    public static void RunLoadTests()
    {
        Console.WriteLine("Starting load tests...");
        
        var memoryPoolScenario = CreateMemoryPoolScenario();
        var resourceQuotaScenario = CreateResourceQuotaScenario();
        var kernelExecutionScenario = CreateKernelExecutionScenario();
        
        NBomberRunner
            .RegisterScenarios(memoryPoolScenario, resourceQuotaScenario, kernelExecutionScenario)
            .Run();
    }
    
    private static ScenarioProps CreateMemoryPoolScenario()
    {
        var poolManager = new MemoryPoolManager(NullLoggerFactory.Instance);
        
        return Scenario.Create("memory_pool_stress", async context =>
        {
            var pool = poolManager.GetPool<float>();
            var size = Random.Shared.Next(1024, 65536);
            
            var memory = pool.Rent(size);
            
            // Simulate some work
            await Task.Delay(Random.Shared.Next(1, 10));
            
            memory.Dispose();
            
            return Response.Ok();
        })
        .WithLoadSimulations(
            Simulation.InjectPerSec(rate: 100, during: TimeSpan.FromSeconds(30)),
            Simulation.KeepConstant(copies: 50, during: TimeSpan.FromSeconds(30)),
            Simulation.InjectPerSec(rate: 200, during: TimeSpan.FromSeconds(30))
        );
    }
    
    private static ScenarioProps CreateResourceQuotaScenario()
    {
        var quotaManager = new ResourceQuotaManager(
            NullLogger<ResourceQuotaManager>.Instance,
            Microsoft.Extensions.Options.Options.Create(new ResourceQuotaOptions()));
        
        return Scenario.Create("resource_quota_stress", async context =>
        {
            var tenantId = $"tenant_{Random.Shared.Next(1, 100)}";
            
            var allocation = await quotaManager.RequestAllocationAsync(
                tenantId,
                new ResourceRequest
                {
                    RequestedMemoryBytes = Random.Shared.Next(1024, 1024 * 1024),
                    RequestedKernels = Random.Shared.Next(1, 5),
                    BatchSize = Random.Shared.Next(10, 100),
                    EstimatedDuration = TimeSpan.FromMilliseconds(Random.Shared.Next(10, 100))
                });
            
            if (allocation == null)
            {
                return Response.Fail("Allocation denied");
            }
            
            // Simulate work
            await Task.Delay(Random.Shared.Next(5, 50));
            
            await quotaManager.ReleaseAllocationAsync(
                tenantId,
                allocation.AllocationId,
                allocation.AllocatedMemoryBytes,
                allocation.AllocatedKernels);
            
            return Response.Ok();
        })
        .WithLoadSimulations(
            Simulation.InjectPerSec(rate: 50, during: TimeSpan.FromSeconds(20)),
            Simulation.InjectPerSecRandom(minRate: 20, maxRate: 100, during: TimeSpan.FromSeconds(30)),
            Simulation.KeepConstant(copies: 100, during: TimeSpan.FromSeconds(20))
        );
    }
    
    private static ScenarioProps CreateKernelExecutionScenario()
    {
        var kernel = new CpuVectorAddKernel();
        
        return Scenario.Create("kernel_execution_stress", async context =>
        {
            var batchSize = Random.Shared.Next(10, 100);
            var batch = new float[batchSize][];
            
            for (int i = 0; i < batchSize; i++)
            {
                batch[i] = new float[1024];
                for (int j = 0; j < 1024; j++)
                {
                    batch[i][j] = Random.Shared.NextSingle();
                }
            }
            
            var handle = await kernel.SubmitBatchAsync(batch);
            
            var results = new List<float>();
            await foreach (var result in kernel.ReadResultsAsync(handle))
            {
                results.Add(result);
            }
            
            if (results.Count == 0)
            {
                return Response.Fail("No results returned");
            }
            
            return Response.Ok(sizeBytes: batchSize * 1024 * sizeof(float));
        })
        .WithLoadSimulations(
            Simulation.InjectPerSec(rate: 20, during: TimeSpan.FromSeconds(30)),
            Simulation.InjectPerSecRandom(minRate: 10, maxRate: 50, during: TimeSpan.FromSeconds(30))
        );
    }
}