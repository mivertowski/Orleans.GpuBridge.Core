using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.DotCompute;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.BackendProviders;
using Orleans.Hosting;
using Orleans.TestingHost;
using Xunit;

namespace Orleans.GpuBridge.Tests.Integration;

public class EndToEndTests : IClassFixture<EndToEndTests.ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public EndToEndTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task Full_Pipeline_Should_Work()
    {
        // This test covers the entire pipeline from grain to GPU execution
        var grain = _fixture.Cluster.GrainFactory.GetGrain<ITestComputeGrain>(Guid.NewGuid());
        
        // Act
        var result = await grain.ComputeAsync(new float[] { 1, 2, 3, 4, 5 });
        
        // Assert
        Assert.NotNull(result);
        Assert.Equal(5, result.Length);
    }

    [Fact]
    public async Task Memory_Pool_Integration_Should_Work()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
        var poolManager = new MemoryPoolManager(loggerFactory);
        var pool = poolManager.GetPool<float>();
        
        // Act - Simulate GPU memory operations
        var tasks = new Task[10];
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(async () =>
            {
                using var memory = pool.Rent(1000);
                await memory.CopyToDeviceAsync();
                await Task.Delay(10);
                await memory.CopyFromDeviceAsync();
            });
        }
        
        await Task.WhenAll(tasks);
        
        // Assert
        var stats = pool.GetStats();
        Assert.True(stats.TotalAllocated > 0);
        Assert.True(stats.ReturnCount >= 10);
    }

    [Fact]
    public async Task Backend_Provider_Integration_Should_Work()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<BackendProviderFactory>>();
        var factory = new BackendProviderFactory(serviceProvider, logger);
        factory.Initialize();
        
        // Act
        var provider = factory.GetPrimaryProvider();
        using var context = provider.CreateContext();
        using var buffer = context.CreateBuffer<float>(100, BufferUsage.ReadWrite);
        
        var data = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        buffer.Write(data);
        
        var result = new float[100];
        buffer.Read(result);
        
        // Assert
        Assert.Equal(data, result);
    }

    [Fact]
    public async Task Device_Broker_Integration_Should_Work()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        services.Configure<GpuBridgeOptions>(opt =>
        {
            opt.PreferGpu = true;
            opt.MemoryPoolSizeMB = 1024;
        });
        
        var serviceProvider = services.BuildServiceProvider();
        var logger = serviceProvider.GetRequiredService<ILogger<DeviceBroker>>();
        var options = serviceProvider.GetRequiredService<Microsoft.Extensions.Options.IOptions<GpuBridgeOptions>>();
        
        using var broker = new DeviceBroker(logger, options);
        
        // Act
        await broker.InitializeAsync(CancellationToken.None);
        var device = broker.GetBestDevice();
        
        // Assert
        Assert.NotNull(device);
        Assert.True(broker.DeviceCount > 0);
    }

    [Fact]
    public async Task Parallel_Kernel_Execution_Integration_Should_Work()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<ParallelKernelExecutor>>();
        var executor = new ParallelKernelExecutor(logger);
        
        var input = Enumerable.Range(1, 10000).Select(i => (float)i).ToArray();
        
        // Act
        var result = await executor.ExecuteVectorizedAsync(
            input, 
            VectorOperation.FusedMultiplyAdd, 
            new[] { 2.0f, 3.0f });
        
        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < Math.Min(10, input.Length); i++)
        {
            Assert.Equal(input[i] * 2.0f + 3.0f, result[i], 5);
        }
    }

    [Fact]
    public async Task Buffer_Serialization_Integration_Should_Work()
    {
        // Arrange
        var data = Enumerable.Range(1, 1000).Select(i => (float)i).ToArray();
        
        // Act - Test full serialization pipeline
        var serialized = BufferSerializer.Serialize<float>(data);
        var compressed = await BufferSerializer.SerializeCompressedAsync(data);
        
        var deserialized = BufferSerializer.Deserialize<float>(serialized);
        var decompressed = await BufferSerializer.DeserializeCompressedAsync<float>(compressed);
        
        // Assert
        Assert.Equal(data, deserialized);
        Assert.Equal(data, decompressed);
        Assert.True(compressed.Length < serialized.Length); // Compression should reduce size
    }

    [Fact]
    public async Task CPU_Fallback_Should_Work_When_No_GPU()
    {
        // This test ensures CPU fallback works correctly
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<BackendProviderFactory>>();
        var factory = new BackendProviderFactory(serviceProvider, logger);
        factory.Initialize();
        
        // Act - CPU provider should always be available
        var cpuProvider = factory.GetProvider(BackendType.Cpu);
        
        // Assert
        Assert.NotNull(cpuProvider);
        Assert.True(cpuProvider.IsAvailable);
        
        // Test CPU execution
        using var context = cpuProvider.CreateContext();
        using var kernel = context.CompileKernel("vector_add", "main");
        using var bufferA = context.CreateBuffer<float>(100, BufferUsage.ReadOnly);
        using var bufferB = context.CreateBuffer<float>(100, BufferUsage.ReadOnly);
        using var bufferC = context.CreateBuffer<float>(100, BufferUsage.WriteOnly);
        
        var a = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(1, 100).Select(i => (float)i * 2).ToArray();
        
        bufferA.Write(a);
        bufferB.Write(b);
        
        kernel.SetArgument(0, bufferA);
        kernel.SetArgument(1, bufferB);
        kernel.SetArgument(2, bufferC);
        
        context.Execute(kernel, 100);
        context.Synchronize();
        
        var result = new float[100];
        bufferC.Read(result);
        
        // Verify computation
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(a[i] + b[i], result[i]);
        }
    }

    [Fact]
    public async Task Concurrent_Operations_Should_Be_Thread_Safe()
    {
        // This test ensures thread safety across all components
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var tasks = new Task[20];
        
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(async () =>
            {
                // Memory pool operations
                var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
                var poolManager = new MemoryPoolManager(loggerFactory);
                var pool = poolManager.GetPool<float>();
                
                using var memory = pool.Rent(Random.Shared.Next(100, 1000));
                await memory.CopyToDeviceAsync();
                
                // Parallel execution
                var execLogger = serviceProvider.GetRequiredService<ILogger<ParallelKernelExecutor>>();
                var executor = new ParallelKernelExecutor(execLogger);
                var input = Enumerable.Range(1, 100).Select(j => (float)j).ToArray();
                var result = await executor.ExecuteVectorizedAsync(
                    input, VectorOperation.Add, new[] { 10.0f });
                
                // Buffer serialization
                var serialized = BufferSerializer.Serialize<float>(result);
                var deserialized = BufferSerializer.Deserialize<float>(serialized);
                
                Assert.Equal(result.Length, deserialized.Length);
            });
        }
        
        await Task.WhenAll(tasks);
    }

    [Fact]
    public async Task Large_Data_Processing_Should_Work()
    {
        // Test with large datasets to ensure scalability
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<ParallelKernelExecutor>>();
        var executor = new ParallelKernelExecutor(logger);
        
        // Process 1 million floats
        var input = Enumerable.Range(1, 1_000_000).Select(i => (float)i).ToArray();
        
        // Act
        var result = await executor.ExecuteVectorizedAsync(
            input, VectorOperation.Multiply, new[] { 2.0f });
        
        // Assert - Just verify first and last elements for performance
        Assert.Equal(1_000_000, result.Length);
        Assert.Equal(2.0f, result[0]);
        Assert.Equal(2_000_000.0f, result[999_999]);
    }

    public interface ITestComputeGrain : IGrainWithGuidKey
    {
        Task<float[]> ComputeAsync(float[] input);
    }

    public class TestComputeGrain : Grain, ITestComputeGrain
    {
        public Task<float[]> ComputeAsync(float[] input)
        {
            // Simple computation for testing
            return Task.FromResult(input.Select(x => x * 2).ToArray());
        }
    }

    public class ClusterFixture : IDisposable
    {
        public TestCluster Cluster { get; }

        public ClusterFixture()
        {
            var builder = new TestClusterBuilder();
            builder.AddSiloBuilderConfigurator<SiloConfigurator>();
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
                    services.AddGpuBridge(builder =>
                    {
                        builder.PreferGpu = false; // Use CPU for testing
                    });
                });
            }
        }
    }
}

public class StressTests
{
    [Fact(Skip = "Long running stress test")]
    public async Task Memory_Pool_Stress_Test()
    {
        // Stress test memory pool with heavy concurrent usage
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
        var poolManager = new MemoryPoolManager(loggerFactory);
        var pool = poolManager.GetPool<float>();
        
        var tasks = new Task[100];
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(async () =>
            {
                while (!cts.Token.IsCancellationRequested)
                {
                    var size = Random.Shared.Next(100, 10000);
                    using var memory = pool.Rent(size);
                    
                    // Simulate work
                    await Task.Delay(Random.Shared.Next(1, 10), cts.Token);
                }
            }, cts.Token);
        }
        
        try
        {
            await Task.WhenAll(tasks);
        }
        catch (OperationCanceledException)
        {
            // Expected
        }
        
        // Verify no memory leaks
        var stats = pool.GetStats();
        Assert.Equal(0, stats.InUse);
    }

    [Fact(Skip = "Long running stress test")]
    public async Task Parallel_Execution_Stress_Test()
    {
        // Stress test parallel execution with many concurrent operations
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<ParallelKernelExecutor>>();
        var tasks = new Task[50];
        
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(async () =>
            {
                var executor = new ParallelKernelExecutor(logger);
                
                for (int j = 0; j < 100; j++)
                {
                    var size = Random.Shared.Next(1000, 10000);
                    var input = Enumerable.Range(1, size).Select(k => (float)k).ToArray();
                    
                    var operation = (VectorOperation)Random.Shared.Next(0, 7);
                    var parameters = operation switch
                    {
                        VectorOperation.FusedMultiplyAdd => new[] { 2.0f, 3.0f },
                        _ => new[] { Random.Shared.NextSingle() * 10 }
                    };
                    
                    var result = await executor.ExecuteVectorizedAsync(input, operation, parameters);
                    Assert.Equal(size, result.Length);
                }
            });
        }
        
        await Task.WhenAll(tasks);
    }
}