using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Tests.TestingFramework;
using Xunit;

namespace Orleans.GpuBridge.Tests.Unit;

/// <summary>
/// Comprehensive tests for kernel execution pipeline with mocked GPU
/// </summary>
public class KernelExecutionPipelineTests : TestFixtureBase
{
    [Fact]
    public async Task Given_Valid_Kernel_When_Execute_Single_Batch_Then_Should_Return_Results()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddGpuBridge()
            .AddKernel(k => k
                .Id("test/simple")
                .Input<float[]>()
                .Output<float>()
                .WithFactory(_ => TestKernelFactory.CreateVectorAddKernel()));

        var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<KernelCatalog>();

        var input = new[] { 1f, 2f, 3f, 4f, 5f };

        // Act
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("test/simple"), provider);
        
        var handle = await kernel.SubmitBatchAsync(new[] { input });
        var results = new List<float>();
        
        await foreach (var result in kernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }

        // Assert
        results.Should().NotBeEmpty();
        results.Should().HaveCount(1);
        results[0].Should().Be(15f); // Sum of 1+2+3+4+5
    }

    [Fact]
    public async Task Given_Multiple_Batches_When_Execute_Concurrently_Then_Should_Handle_All_Batches()
    {
        // Arrange
        var kernel = TestKernelFactory.CreateVectorAddKernel();
        var batches = new[]
        {
            new[] { new[] { 1f, 2f, 3f } },
            new[] { new[] { 4f, 5f, 6f } },
            new[] { new[] { 7f, 8f, 9f } }
        };

        // Act
        var tasks = batches.Select(async batch =>
        {
            var handle = await kernel.SubmitBatchAsync(batch);
            var results = new List<float>();
            await foreach (var result in kernel.ReadResultsAsync(handle))
            {
                results.Add(result);
            }
            return results;
        }).ToArray();

        var allResults = await Task.WhenAll(tasks);

        // Assert
        allResults.Should().HaveCount(3);
        allResults[0].Should().Equal(6f);   // 1+2+3
        allResults[1].Should().Equal(15f);  // 4+5+6
        allResults[2].Should().Equal(24f);  // 7+8+9
    }

    [Fact]
    public async Task Given_Large_Batch_When_Execute_Then_Should_Handle_Large_Dataset()
    {
        // Arrange
        var kernel = TestKernelFactory.CreateVectorAddKernel();
        var largeInput = Enumerable.Range(1, 10000).Select(i => (float)i).ToArray();
        var batches = new[] { largeInput };

        // Act
        var handle = await kernel.SubmitBatchAsync(batches);
        var results = new List<float>();
        
        await foreach (var result in kernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }

        // Assert
        results.Should().NotBeEmpty();
        results[0].Should().Be(50005000f); // Sum of 1 to 10000
    }

    [Fact]
    public async Task Given_Kernel_With_Execution_Delay_When_Execute_Then_Should_Respect_Timing()
    {
        // Arrange
        var delay = TimeSpan.FromMilliseconds(100);
        var kernel = TestKernelFactory.CreateSlowKernel<float[], float>(delay);
        var input = new[] { 1f, 2f, 3f };

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Act
        var handle = await kernel.SubmitBatchAsync(new[] { input });
        await foreach (var _ in kernel.ReadResultsAsync(handle))
        {
            // Consume results
        }

        stopwatch.Stop();

        // Assert
        stopwatch.ElapsedMilliseconds.Should().BeGreaterThanOrEqualTo((long)delay.TotalMilliseconds);
    }

    [Fact]
    public async Task Given_Failing_Kernel_When_Execute_Then_Should_Propagate_Exception()
    {
        // Arrange
        var expectedException = new InvalidOperationException("Test kernel failure");
        var kernel = TestKernelFactory.CreateFailingKernel<float[], float>(expectedException);
        var input = new[] { 1f, 2f, 3f };

        // Act & Assert
        var act = async () => await kernel.SubmitBatchAsync(new[] { input });
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("Test kernel failure");
    }

    [Fact]
    public async Task Given_Kernel_With_Invalid_Handle_When_ReadResults_Then_Should_Throw()
    {
        // Arrange
        var kernel = TestKernelFactory.CreateVectorAddKernel();
        var invalidHandle = KernelHandle.Create(); // Not submitted

        // Act & Assert
        var act = async () => await kernel.ReadResultsAsync(invalidHandle).ToListAsync();
        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*Invalid or unknown handle*");
    }

    [Fact]
    public async Task Given_Cancelled_Token_When_Execute_Then_Should_Cancel_Execution()
    {
        // Arrange
        var kernel = TestKernelFactory.CreateSlowKernel<float[], float>(TimeSpan.FromSeconds(5));
        var input = new[] { 1f, 2f, 3f };
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));

        // Act & Assert
        var act = async () => await kernel.SubmitBatchAsync(new[] { input }, ct: cts.Token);
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task Given_Kernel_With_GPU_Hints_When_Execute_Then_Should_Accept_Hints()
    {
        // Arrange
        var kernel = TestKernelFactory.CreateVectorAddKernel();
        var input = new[] { 1f, 2f, 3f };
        var hints = new GpuExecutionHints(
            PreferredDevice: null,
            HighPriority: false,
            MaxMicroBatch: 1024,
            Persistent: true,
            PreferGpu: true,
            Timeout: TimeSpan.FromMilliseconds(30000),
            MaxRetries: null);

        // Act
        var handle = await kernel.SubmitBatchAsync(new[] { input }, hints);
        var results = new List<float>();
        
        await foreach (var result in kernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }

        // Assert
        results.Should().NotBeEmpty();
        handle.Should().NotBeNull();
    }

    [Fact]
    public async Task Given_Multiple_Kernels_When_Execute_Simultaneously_Then_Should_Isolate_Execution()
    {
        // Arrange
        var kernel1 = TestKernelFactory.CreateVectorAddKernel();
        var kernel2 = TestKernelFactory.CreateVectorMultiplyKernel();
        
        var input1 = new[] { 1f, 2f, 3f };
        var input2 = new[] { 4f, 5f, 6f };

        // Act
        var task1 = Task.Run(async () =>
        {
            var handle = await kernel1.SubmitBatchAsync(new[] { input1 });
            var results = new List<float>();
            await foreach (var result in kernel1.ReadResultsAsync(handle))
            {
                results.Add(result);
            }
            return results;
        });

        var task2 = Task.Run(async () =>
        {
            var handle = await kernel2.SubmitBatchAsync(new[] { input2 });
            var results = new List<float[]>();
            await foreach (var result in kernel2.ReadResultsAsync(handle))
            {
                results.Add(result);
            }
            return results;
        });

        var results1 = await task1;
        var results2 = await task2;

        // Assert
        results1.Should().Equal(new[] { 6f }); // Sum: 1+2+3
        results2[0].Should().Equal(new[] { 8f, 10f, 12f }); // Multiply by 2: 4*2, 5*2, 6*2
    }

    [Fact]
    public async Task Given_Kernel_Info_When_GetInfo_Then_Should_Return_Valid_Information()
    {
        // Arrange
        var kernel = TestKernelFactory.CreateVectorAddKernel();

        // Act
        var info = await kernel.GetInfoAsync();

        // Assert
        info.Should().NotBeNull();
        info.Id.Value.Should().Be("test/vector-add");
        info.Description.Should().Be("Test Vector Add");
        info.InputType.Should().Be(typeof(float[]));
        info.OutputType.Should().Be(typeof(float));
        info.SupportsGpu.Should().BeTrue();
        info.PreferredBatchSize.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task Given_Configurable_Kernel_When_Configure_Behavior_Then_Should_Respect_Configuration()
    {
        // Arrange
        var info = TestDataBuilders.KernelInfo()
            .WithId("test/configurable")
            .WithInputType<float[]>()
            .WithOutputType<float>()
            .Build();

        var config = new TestKernelConfiguration
        {
            SubmissionDelay = TimeSpan.FromMilliseconds(50),
            ExecutionDelay = TimeSpan.FromMilliseconds(25),
            ResultCount = 3
        };

        var kernel = new ConfigurableTestKernel<float[], float>(info, config);
        var input = new[] { 1f, 2f, 3f };

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Act
        var handle = await kernel.SubmitBatchAsync(new[] { input });
        var results = new List<float>();
        
        await foreach (var result in kernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }

        stopwatch.Stop();

        // Assert
        results.Should().HaveCount(3, "should return configured number of results");
        stopwatch.ElapsedMilliseconds.Should().BeGreaterThanOrEqualTo(75, "should respect timing delays");
    }

    [Fact]
    public async Task Given_Kernel_With_Failure_Rate_When_Execute_Multiple_Times_Then_Should_Simulate_Failures()
    {
        // Arrange
        var info = TestDataBuilders.KernelInfo()
            .WithId("test/unreliable")
            .WithInputType<float[]>()
            .WithOutputType<float>()
            .Build();

        var config = new TestKernelConfiguration
        {
            ExecutionFailureRate = 0.5 // 50% failure rate
        };

        var kernel = new ConfigurableTestKernel<float[], float>(info, config);
        var input = new[] { 1f, 2f, 3f };

        var successCount = 0;
        var failureCount = 0;

        // Act - Execute multiple times to test failure rate
        for (int i = 0; i < 20; i++)
        {
            try
            {
                var handle = await kernel.SubmitBatchAsync(new[] { input });
                await foreach (var _ in kernel.ReadResultsAsync(handle))
                {
                    // Consume results
                }
                successCount++;
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("Simulated execution failure"))
            {
                failureCount++;
            }
        }

        // Assert
        failureCount.Should().BeGreaterThan(0, "should have some failures with 50% failure rate");
        successCount.Should().BeGreaterThan(0, "should have some successes with 50% failure rate");
        (successCount + failureCount).Should().Be(20, "all executions should be accounted for");
    }

    [Fact]
    public async Task Given_Kernel_Catalog_When_Resolve_Unknown_Kernel_Then_Should_Return_CPU_Fallback()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddGpuBridge(); // No kernels registered
        
        var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<KernelCatalog>();

        // Act
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("unknown/kernel"), provider);

        // Assert
        kernel.Should().NotBeNull();
        var info = await kernel.GetInfoAsync();
        info.SupportsGpu.Should().BeFalse("fallback should be CPU-only");
        info.InputType.Should().Be(typeof(float[]));
        info.OutputType.Should().Be(typeof(float));
    }

    [Fact]
    public async Task Given_Pipeline_With_Memory_Pressure_When_Execute_Large_Batches_Then_Should_Handle_Gracefully()
    {
        // Arrange
        var kernel = TestKernelFactory.CreateVectorAddKernel();
        var largeBatches = Enumerable.Range(0, 100)
            .Select(i => Enumerable.Range(i * 1000, 1000).Select(j => (float)j).ToArray())
            .ToArray();

        // Act
        var tasks = largeBatches.Select(async batch =>
        {
            var handle = await kernel.SubmitBatchAsync(new[] { batch });
            var results = new List<float>();
            await foreach (var result in kernel.ReadResultsAsync(handle))
            {
                results.Add(result);
            }
            return results.Sum();
        });

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().HaveCount(100);
        results.Should().OnlyContain(r => r > 0, "all results should be positive");
    }
}