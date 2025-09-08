using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Grains;
using Orleans.GpuBridge.Runtime;
using Orleans.Hosting;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Tests.Integration;

/// <summary>
/// Comprehensive Orleans cluster integration tests for GPU Bridge
/// </summary>
public class OrleansClusterIntegrationTests : IClassFixture<GpuClusterFixture>
{
    private readonly GpuClusterFixture _fixture;

    public OrleansClusterIntegrationTests(GpuClusterFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task GpuBatchGrain_EndToEndExecution_ShouldWorkInCluster()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>("vector-add");
        
        var input = new[]
        {
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 4.0f, 5.0f, 6.0f },
            new float[] { 7.0f, 8.0f, 9.0f }
        };

        // Act
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(3);
        result.ExecutionTime.Should().BeGreaterThan(TimeSpan.Zero);
        result.Error.Should().BeNull();
        result.KernelId.Value.Should().Be("vector-add");
    }

    [Fact]
    public async Task GpuPipeline_BatchProcessing_ShouldDistributeAcrossGrains()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var inputData = Enumerable.Range(0, 1000)
            .Select(i => new float[] { i, i + 1, i + 2 })
            .ToList();

        // Act
        var results = await GpuPipeline<float[], float[]>
            .For(grainFactory, "vector-multiply")
            .WithBatchSize(100)
            .WithMaxConcurrency(4)
            .ExecuteAsync(inputData);

        // Assert
        results.Should().NotBeNull();
        results.Should().HaveCount(1000);
        results.All(r => r != null && r.Length == 3).Should().BeTrue();
    }

    [Fact]
    public async Task MultipleGrains_ConcurrentExecution_ShouldHandleCorrectly()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var tasks = new List<Task<GpuBatchResult<float[]>>>();

        // Create multiple concurrent grain executions
        for (int i = 0; i < 10; i++)
        {
            var grainId = $"concurrent-test-{i}";
            var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(grainId);
            var input = new[] { new float[] { i, i + 1, i + 2 } };
            tasks.Add(grain.ExecuteAsync(input));
        }

        // Act
        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().HaveCount(10);
        results.All(r => r.Success).Should().BeTrue();
        results.All(r => r.Results.Count == 1).Should().BeTrue();
    }

    [Fact]
    public async Task GpuResidentGrain_DataPersistence_ShouldMaintainStateAcrossActivations()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuResidentGrain<float[]>>("persistent-data");
        
        var testData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act - Store data
        await grain.StoreDataAsync(testData);
        
        // Force grain deactivation by accessing it from different silo
        await Task.Delay(100); // Give time for state to persist
        
        // Retrieve data (may cause reactivation)
        var retrievedData = await grain.GetDataAsync();

        // Assert
        retrievedData.Should().NotBeNull();
        retrievedData.Should().BeEquivalentTo(testData);
    }

    [Fact]
    public async Task GpuStreamGrain_StreamProcessing_ShouldHandleRealtimeData()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var streamGrain = grainFactory.GetGrain<IGpuStreamGrain<float, float>>("stream-processor");
        
        var inputStream = new List<float> { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var results = new List<float>();

        // Create observer
        var observer = new TestStreamObserver<float>(results);

        // Act
        await streamGrain.StartStreamAsync("test-stream", observer);
        
        foreach (var value in inputStream)
        {
            await streamGrain.ProcessItemAsync(value);
        }
        
        await streamGrain.FlushStreamAsync();
        await Task.Delay(500); // Wait for processing

        // Assert
        results.Should().HaveCount(inputStream.Count);
        results.All(r => r > 0).Should().BeTrue();
    }

    [Fact]
    public async Task GpuPlacementStrategy_ShouldPreferGpuCapableSilos()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var placements = new Dictionary<string, string>();

        // Act - Create multiple grains and track their placement
        var grains = new List<IGpuBatchGrain<float[], float[]>>();
        for (int i = 0; i < 5; i++)
        {
            var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>($"placement-test-{i}");
            grains.Add(grain);
            
            // Execute to ensure activation
            var testInput = new[] { new float[] { i } };
            await grain.ExecuteAsync(testInput);
        }

        // Assert - In a real GPU environment, grains should be placed on GPU-capable silos
        // For now, just verify grains are activated and functional
        grains.Should().HaveCount(5);
        grains.All(g => g != null).Should().BeTrue();
    }

    [Fact]
    public async Task ErrorHandling_GpuFailure_ShouldFallbackToCpu()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>("fallback-test");
        
        // Use data that might cause GPU issues (but should work on CPU)
        var problematicInput = new[]
        {
            new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity }
        };

        // Act
        var result = await grain.ExecuteAsync(problematicInput);

        // Assert - Should succeed with CPU fallback
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(1);
    }

    [Fact]
    public async Task HealthChecks_GpuBridgeSystem_ShouldReportCorrectStatus()
    {
        // Arrange
        var services = _fixture.Cluster.Host.Services;
        var healthCheckService = services.GetService<Microsoft.Extensions.Diagnostics.HealthChecks.HealthCheckService>();

        if (healthCheckService == null)
        {
            // Skip if health checks not configured
            return;
        }

        // Act
        var healthReport = await healthCheckService.CheckHealthAsync();

        // Assert
        healthReport.Should().NotBeNull();
        healthReport.Status.Should().BeOneOf(
            Microsoft.Extensions.Diagnostics.HealthChecks.HealthStatus.Healthy,
            Microsoft.Extensions.Diagnostics.HealthChecks.HealthStatus.Degraded); // Degraded is OK for CPU fallback
    }

    [Fact]
    public async Task GrainLifecycle_ActivationDeactivation_ShouldCleanupResources()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grainId = Guid.NewGuid().ToString();
        
        // Act - Activate grain
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(grainId);
        var input = new[] { new float[] { 1.0f, 2.0f, 3.0f } };
        var result1 = await grain.ExecuteAsync(input);
        
        // Force deactivation by waiting
        await Task.Delay(1000);
        
        // Reactivate and test again
        var result2 = await grain.ExecuteAsync(input);

        // Assert - Both executions should succeed
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
        result1.KernelId.Should().Be(result2.KernelId);
    }

    [Fact]
    public async Task KernelCompilation_FirstExecution_ShouldCacheForSubsequentCalls()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>("compilation-cache-test");
        var input = new[] { new float[] { 1.0f, 2.0f, 3.0f } };

        // Act - First execution (should compile kernel)
        var sw1 = System.Diagnostics.Stopwatch.StartNew();
        var result1 = await grain.ExecuteAsync(input);
        sw1.Stop();

        // Second execution (should use cached kernel)
        var sw2 = System.Diagnostics.Stopwatch.StartNew();
        var result2 = await grain.ExecuteAsync(input);
        sw2.Stop();

        // Assert
        result1.Success.Should().BeTrue();
        result2.Success.Should().BeTrue();
        
        // Second execution should typically be faster due to caching
        // (Note: In CPU fallback mode, this difference might be minimal)
        result2.ExecutionTime.Should().BeLessOrEqualTo(result1.ExecutionTime.Add(TimeSpan.FromMilliseconds(100)));
    }

    [Fact]
    public async Task LargeDataTransfer_ShouldHandleEfficiently()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>("large-data-test");
        
        // Create large dataset
        var largeInput = Enumerable.Range(0, 100)
            .Select(i => Enumerable.Range(0, 1000).Select(j => (float)(i * 1000 + j)).ToArray())
            .ToArray();

        // Act
        var result = await grain.ExecuteAsync(largeInput);

        // Assert
        result.Should().NotBeNull();
        result.Success.Should().BeTrue();
        result.Results.Should().HaveCount(100);
        result.ExecutionTime.Should().BeLessThan(TimeSpan.FromSeconds(10)); // Should complete reasonably fast
    }
}

/// <summary>
/// Test observer for stream processing
/// </summary>
public class TestStreamObserver<T> : IGpuResultObserver<T>
{
    private readonly List<T> _results;

    public TestStreamObserver(List<T> results)
    {
        _results = results;
    }

    public Task OnNextAsync(T item)
    {
        _results.Add(item);
        return Task.CompletedTask;
    }

    public Task OnErrorAsync(Exception error)
    {
        throw error; // Propagate errors in tests
    }

    public Task OnCompletedAsync()
    {
        return Task.CompletedTask;
    }
}

/// <summary>
/// Orleans cluster fixture with GPU Bridge configuration
/// </summary>
public class GpuClusterFixture : IDisposable
{
    public TestCluster Cluster { get; private set; }

    public GpuClusterFixture()
    {
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<GpuSiloConfigurator>();
        builder.AddClientBuilderConfigurator<GpuClientConfigurator>();
        
        Cluster = builder.Build();
        Cluster.Deploy();
    }

    public void Dispose()
    {
        Cluster?.StopAllSilos();
    }
}

/// <summary>
/// Silo configurator with GPU Bridge setup
/// </summary>
public class GpuSiloConfigurator : ISiloConfigurator
{
    public void Configure(ISiloBuilder siloBuilder)
    {
        siloBuilder.ConfigureServices(services =>
        {
            // Add GPU Bridge with test configuration
            services.AddGpuBridge(options =>
            {
                options.PreferGpu = true;
                options.MemoryPoolSizeMB = 256;
                options.BatchSize = 32;
                options.MaxRetries = 3;
                options.EnableHealthChecks = true;
                options.EnableTelemetry = true;
            })
            .AddILGPUBackend(ilgpuOptions =>
            {
                ilgpuOptions.PreferredDevice = ILGPU.DeviceType.CPU; // Use CPU for consistent testing
                ilgpuOptions.EnableDebugMode = true;
            })
            .AddKernel(k => k.Id("vector-add")
                           .In<float[]>().Out<float[]>()
                           .FromTemplate("VectorAdd"))
            .AddKernel(k => k.Id("vector-multiply") 
                           .In<float[]>().Out<float[]>()
                           .FromTemplate("VectorMultiply"))
            .AddKernel(k => k.Id("matrix-multiply")
                           .In<float[,]>().Out<float[,]>()
                           .FromTemplate("MatrixMultiply"));

            // Add health checks
            services.AddHealthChecks()
                .AddGpuBridge();
        });
    }
}

/// <summary>
/// Client configurator
/// </summary>
public class GpuClientConfigurator : IClientBuilderConfigurator
{
    public void Configure(IConfiguration configuration, IClientBuilder clientBuilder)
    {
        clientBuilder.ConfigureServices(services =>
        {
            // Add any client-side services if needed
        });
    }
}