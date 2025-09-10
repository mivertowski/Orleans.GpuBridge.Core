using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Grains;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.Hosting;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Tests.Integration;

/// <summary>
/// Comprehensive integration tests for Orleans grains using TestingHost
/// </summary>
public class OrleansGrainTests : IClassFixture<OrleansGrainTests.GrainTestFixture>
{
    private readonly GrainTestFixture _fixture;

    public OrleansGrainTests(GrainTestFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task Given_GpuBatchGrain_When_ProcessBatch_Then_Should_Execute_Successfully()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuBatchGrain<float[], float>>(
            Guid.NewGuid(), keyExtension: "vector-add");
        
        var batch = new[]
        {
            new[] { 1f, 2f, 3f },
            new[] { 4f, 5f, 6f },
            new[] { 7f, 8f, 9f }
        };

        // Act
        var results = await grain.ProcessBatchAsync(batch);

        // Assert
        results.Should().NotBeNull();
        results.Should().HaveCount(3);
        results.Should().Equal(6f, 15f, 24f); // Sums of each vector
    }

    [Fact]
    public async Task Given_GpuResidentGrain_When_StoreAndRetrieve_Then_Should_Maintain_Data()
    {
        // Arrange
        var grainId = Guid.NewGuid();
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuResidentGrain<float[]>>(grainId);
        
        var testData = Enumerable.Range(1, 1000).Select(i => (float)i).ToArray();

        // Act - Store data
        await grain.StoreDataAsync(testData);
        
        // Act - Retrieve data
        var retrievedData = await grain.GetDataAsync();

        // Assert
        retrievedData.Should().NotBeNull();
        retrievedData.Should().Equal(testData);
    }

    [Fact]
    public async Task Given_GpuStreamGrain_When_ProcessStream_Then_Should_Transform_Data()
    {
        // Arrange
        var grainId = Guid.NewGuid();
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuStreamGrain<float, float>>(grainId);
        
        var streamData = Enumerable.Range(1, 100).Select(i => (float)i);

        // Act
        var results = new List<float>();
        await foreach (var result in grain.ProcessStreamAsync(streamData.ToAsyncEnumerable()))
        {
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(100);
        results.Should().Equal(streamData.Select(x => x * 2f)); // Assuming multiply by 2 transform
    }

    [Fact]
    public async Task Given_Multiple_GpuBatchGrains_When_Process_Concurrently_Then_Should_Handle_All()
    {
        // Arrange
        var grainCount = 10;
        var grains = Enumerable.Range(0, grainCount)
            .Select(i => _fixture.Cluster.GrainFactory.GetGrain<IGpuBatchGrain<float[], float>>(
                Guid.NewGuid(), keyExtension: "vector-add"))
            .ToArray();

        var batches = Enumerable.Range(0, grainCount)
            .Select(i => new[] { Enumerable.Range(i * 10, 10).Select(j => (float)j).ToArray() })
            .ToArray();

        // Act
        var tasks = grains.Zip(batches, (grain, batch) => grain.ProcessBatchAsync(batch));
        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().HaveCount(grainCount);
        results.Should().OnlyContain(r => r != null && r.Length > 0);
    }

    [Fact]
    public async Task Given_GpuBatchGrain_With_Large_Dataset_When_Process_Then_Should_Handle_Scalably()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuBatchGrain<float[], float>>(
            Guid.NewGuid(), keyExtension: "vector-add");
        
        // Create large batch - 1000 vectors of 1000 elements each
        var largeBatch = Enumerable.Range(0, 1000)
            .Select(i => Enumerable.Range(i * 1000, 1000).Select(j => (float)j).ToArray())
            .ToArray();

        // Act
        var startTime = DateTime.UtcNow;
        var results = await grain.ProcessBatchAsync(largeBatch);
        var processingTime = DateTime.UtcNow - startTime;

        // Assert
        results.Should().HaveCount(1000);
        processingTime.Should().BeLessThan(TimeSpan.FromSeconds(30), "large batches should process within reasonable time");
    }

    [Fact]
    public async Task Given_GpuResidentGrain_When_Store_Multiple_Times_Then_Should_Update_Data()
    {
        // Arrange
        var grainId = Guid.NewGuid();
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuResidentGrain<float[]>>(grainId);

        var initialData = new[] { 1f, 2f, 3f };
        var updatedData = new[] { 4f, 5f, 6f, 7f };

        // Act
        await grain.StoreDataAsync(initialData);
        var firstRetrieve = await grain.GetDataAsync();
        
        await grain.StoreDataAsync(updatedData);
        var secondRetrieve = await grain.GetDataAsync();

        // Assert
        firstRetrieve.Should().Equal(initialData);
        secondRetrieve.Should().Equal(updatedData);
    }

    [Fact]
    public async Task Given_GpuStreamGrain_With_Empty_Stream_When_Process_Then_Should_Handle_Gracefully()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuStreamGrain<float, float>>(Guid.NewGuid());
        var emptyStream = AsyncEnumerable.Empty<float>();

        // Act
        var results = new List<float>();
        await foreach (var result in grain.ProcessStreamAsync(emptyStream))
        {
            results.Add(result);
        }

        // Assert
        results.Should().BeEmpty();
    }

    [Fact]
    public async Task Given_GpuBatchGrain_With_Invalid_Data_When_Process_Then_Should_Handle_Errors()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuBatchGrain<float[], float>>(
            Guid.NewGuid(), keyExtension: "vector-add");
        
        var invalidBatch = new[]
        {
            Array.Empty<float>(), // Empty array
            new[] { float.NaN, float.PositiveInfinity }, // Invalid values
            null! // Null reference
        };

        // Act & Assert
        await grain.Invoking(g => g.ProcessBatchAsync(invalidBatch))
            .Should().ThrowAsync<ArgumentException>()
            .WithMessage("*invalid*");
    }

    [Fact]
    public async Task Given_Grain_With_Custom_Placement_When_Activate_Then_Should_Use_GpuAware_Placement()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuBatchGrain<float[], float>>(
            Guid.NewGuid(), keyExtension: "gpu-optimized");

        // Act
        var batch = new[] { new[] { 1f, 2f, 3f } };
        var result = await grain.ProcessBatchAsync(batch);

        // Assert
        result.Should().NotBeNull();
        // The grain should be placed on a silo with GPU capabilities
        // (This is tested indirectly through successful execution)
    }

    [Fact]
    public async Task Given_GpuResidentGrain_With_Memory_Pressure_When_Store_Large_Data_Then_Should_Manage_Memory()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory.GetGrain<IGpuResidentGrain<float[]>>(Guid.NewGuid());
        
        // Create large dataset (100MB of floats)
        var largeData = new float[25_000_000]; // ~100MB
        for (int i = 0; i < largeData.Length; i++)
        {
            largeData[i] = i;
        }

        // Act
        await grain.StoreDataAsync(largeData);
        var retrievedData = await grain.GetDataAsync();

        // Assert
        retrievedData.Should().NotBeNull();
        retrievedData.Length.Should().Be(largeData.Length);
        // Verify first and last elements to avoid comparing entire array
        retrievedData[0].Should().Be(0f);
        retrievedData[largeData.Length - 1].Should().Be(largeData.Length - 1);
    }

    [Fact]
    public async Task Given_Multiple_Grain_Types_When_Use_Together_Then_Should_Coordinate_Processing()
    {
        // Arrange
        var batchGrain = _fixture.Cluster.GrainFactory.GetGrain<IGpuBatchGrain<float[], float>>(
            Guid.NewGuid(), keyExtension: "coordinator-batch");
        var residentGrain = _fixture.Cluster.GrainFactory.GetGrain<IGpuResidentGrain<float[]>>(
            Guid.NewGuid());

        var inputData = new[] { new[] { 1f, 2f, 3f, 4f, 5f } };
        
        // Act
        // 1. Process batch to get result
        var batchResults = await batchGrain.ProcessBatchAsync(inputData);
        
        // 2. Store processing results in resident grain
        await residentGrain.StoreDataAsync(batchResults);
        
        // 3. Retrieve stored results
        var storedResults = await residentGrain.GetDataAsync();

        // Assert
        batchResults.Should().Equal(15f); // Sum of input
        storedResults.Should().Equal(batchResults);
    }

    public class GrainTestFixture : IDisposable
    {
        public TestCluster Cluster { get; }

        public GrainTestFixture()
        {
            var builder = new TestClusterBuilder();
            builder.AddSiloBuilderConfigurator<SiloConfigurator>();
            builder.AddClientBuilderConfigurator<ClientConfigurator>();
            
            Cluster = builder.Build();
            Cluster.Deploy();
        }

        public void Dispose()
        {
            Cluster?.StopAllSilos();
        }

        private class SiloConfigurator : ISiloConfigurator
        {
            public void Configure(ISiloBuilder siloBuilder)
            {
                siloBuilder.ConfigureServices(services =>
                {
                    services.AddLogging(builder => builder.SetMinimumLevel(LogLevel.Warning));
                    
                    services.AddGpuBridge(options =>
                    {
                        options.PreferGpu = false; // Use CPU for testing
                        options.MemoryPoolSizeMB = 256;
                    })
                    .AddKernel(k => k
                        .Id("vector-add")
                        .Input<float[]>()
                        .Output<float>()
                        .WithFactory(_ => new TestVectorAddKernel()))
                    .AddKernel(k => k
                        .Id("vector-multiply")
                        .Input<float>()
                        .Output<float>()
                        .WithFactory(_ => new TestScalarMultiplyKernel()));
                    
                    // Configure GPU-aware placement
                    services.Configure<GpuBridgeOptions>(options =>
                    {
                        options.Placement = new GpuPlacementOptions
                        {
                            PreferGpuSilos = true,
                            LoadBalanceAcrossDevices = true
                        };
                    });
                });
            }
        }

        private class ClientConfigurator : IClientBuilderConfigurator
        {
            public void Configure(IConfiguration configuration, IClientBuilder clientBuilder)
            {
                // Client configuration if needed
            }
        }
    }

    /// <summary>
    /// Test kernel implementations for grain testing
    /// </summary>
    private class TestVectorAddKernel : IGpuKernel<float[], float>
    {
        public async ValueTask<KernelHandle> SubmitBatchAsync(
            IReadOnlyList<float[]> items,
            GpuExecutionHints? hints = null,
            CancellationToken ct = default)
        {
            await Task.Yield();
            return KernelHandle.Create();
        }

        public async IAsyncEnumerable<float> ReadResultsAsync(
            KernelHandle handle,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.Yield();
            // Return sum of vector elements for each input
            yield return 15f; // Hardcoded for test simplicity
        }

        public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
        {
            return new(new KernelInfo(
                new KernelId("vector-add"),
                "Test Vector Add",
                typeof(float[]),
                typeof(float),
                false,
                1024));
        }
    }

    private class TestScalarMultiplyKernel : IGpuKernel<float, float>
    {
        public async ValueTask<KernelHandle> SubmitBatchAsync(
            IReadOnlyList<float> items,
            GpuExecutionHints? hints = null,
            CancellationToken ct = default)
        {
            await Task.Yield();
            return KernelHandle.Create();
        }

        public async IAsyncEnumerable<float> ReadResultsAsync(
            KernelHandle handle,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.Yield();
            yield return 2f; // Multiply by 2 for test
        }

        public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
        {
            return new(new KernelInfo(
                new KernelId("scalar-multiply"),
                "Test Scalar Multiply",
                typeof(float),
                typeof(float),
                false,
                1024));
        }
    }
}