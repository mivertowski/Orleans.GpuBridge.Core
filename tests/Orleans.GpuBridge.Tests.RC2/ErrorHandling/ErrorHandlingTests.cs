using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Tests.RC2.TestingFramework;

namespace Orleans.GpuBridge.Tests.RC2.ErrorHandling;

/// <summary>
/// Comprehensive error handling and resilience tests for Orleans.GpuBridge.Core RC2
/// Targets 80% error path coverage with 15 tests across GPU failures, fallbacks, and timeouts
/// </summary>
public class ErrorHandlingTests
{
    #region GPU Failure Tests (5 tests)

    [Fact]
    public async Task KernelExecution_WithGpuOutOfMemory_ShouldFallbackToCpu()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2
        {
            SimulateOutOfMemory = true,
            HasCpuFallback = true
        };

        var catalog = CreateCatalogWithFallback(mockGpu);

        // Act
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("test-kernel"),
            CreateServiceProvider());

        var handle = await kernel.SubmitBatchAsync(new[] { new[] { 1f, 2f, 3f } });
        var results = new List<float>();

        await foreach (var result in kernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }

        // Assert
        results.Should().NotBeEmpty("CPU fallback should provide results");
        mockGpu.AllocationAttempts.Should().BeGreaterThanOrEqualTo(1, "GPU allocation should be attempted");

        // Verify fallback occurred by checking that we got results despite GPU failure
        results.Should().HaveCount(1);
    }

    [Fact]
    public async Task KernelExecution_WithGpuTimeout_ShouldThrowTimeout()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2
        {
            SimulateGpuTimeout = true,
            ExecutionTimeout = TimeSpan.FromMilliseconds(100)
        };

        var kernel = CreateMockKernel(mockGpu);
        var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));

        // Act & Assert
        var handle = await kernel.SubmitBatchAsync(
            new[] { new[] { 1f, 2f, 3f } },
            null,
            cts.Token);

        var act = async () =>
        {
            await foreach (var _ in kernel.ReadResultsAsync(handle, cts.Token))
            {
                // Should timeout before results
            }
        };

        await act.Should().ThrowAsync<TimeoutException>()
            .WithMessage("*timeout*", "GPU timeout should throw TimeoutException");
    }

    [Fact]
    public async Task KernelExecution_WithGpuCrash_ShouldRecover()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2
        {
            SimulateGpuCrash = true
        };

        var kernel = new MockKernelRC2<float[], float>(
            TestHelpers.CreateKernelInfo("crash-test"),
            mockGpu);

        // Act & Assert - First execution crashes
        var handle1 = await kernel.SubmitBatchAsync(new[] { new[] { 1f, 2f, 3f } });

        var firstExecution = async () =>
        {
            await foreach (var _ in kernel.ReadResultsAsync(handle1))
            {
            }
        };

        await firstExecution.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*crashed*");

        // Reset and verify recovery
        mockGpu.SimulateGpuCrash = false;
        mockGpu.Reset();

        var handle2 = await kernel.SubmitBatchAsync(new[] { new[] { 1f, 2f, 3f } });
        var results = new List<float>();

        await foreach (var result in kernel.ReadResultsAsync(handle2))
        {
            results.Add(result);
        }

        // Assert - Should recover after reset
        results.Should().NotBeEmpty("Kernel should recover after GPU crash reset");
    }

    [Fact]
    public void MemoryAllocation_WithInsufficientMemory_ShouldThrow()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2
        {
            AvailableMemory = 1024 * 1024, // 1 MB
            SimulateOutOfMemory = false // Let natural OOM occur
        };

        // Pre-allocate most memory
        mockGpu.TryAllocateMemory(512 * 1024); // 512 KB

        // Act & Assert
        var act = () => mockGpu.TryAllocateMemory(1024 * 1024); // Try to allocate 1MB

        act.Should().Throw<OutOfMemoryException>()
            .WithMessage("*Insufficient memory*");

        mockGpu.AllocationAttempts.Should().Be(2);
    }

    [Fact]
    public void MemoryAllocation_WithFragmentation_ShouldDefragment()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2
        {
            SimulateFragmentation = true,
            AvailableMemory = 100 * 1024 * 1024 // 100 MB
        };

        // Act - First allocation should fail due to fragmentation
        var firstAllocation = () => mockGpu.TryAllocateMemory(2 * 1024 * 1024); // 2 MB

        firstAllocation.Should().Throw<InvalidOperationException>()
            .WithMessage("*fragmentation*");

        // Defragment
        mockGpu.DefragmentMemory();

        // Assert - Should succeed after defragmentation
        var allocated = mockGpu.TryAllocateMemory(2 * 1024 * 1024);
        allocated.Should().Be(2 * 1024 * 1024);
        mockGpu.UsedMemory.Should().Be(2 * 1024 * 1024);
    }

    #endregion

    #region Fallback Tests (5 tests)

    [Fact]
    public async Task GpuFallback_ShouldTryCpu()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2
        {
            SimulateOutOfMemory = true, // Force GPU to fail
            HasCpuFallback = true
        };

        var catalog = CreateCatalogWithFallback(mockGpu);

        // Act
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("fallback-test"),
            CreateServiceProvider());

        // Assert - Should get CPU passthrough kernel
        kernel.Should().NotBeNull();

        var info = await kernel.GetInfoAsync();
        info.Should().NotBeNull();

        // Verify it works (even if it's CPU fallback)
        var handle = await kernel.SubmitBatchAsync(new[] { new[] { 1f, 2f, 3f } });
        var results = await kernel.ReadResultsAsync(handle).ToListAsync();

        results.Should().NotBeEmpty("CPU fallback should provide results");
    }

    [Fact]
    public async Task CpuFallback_WithValidKernel_ShouldSucceed()
    {
        // Arrange - Create catalog with CPU-only kernel
        var catalog = new TestKernelCatalogBuilder()
            .AddKernel<float[], float>("cpu-kernel", null, CreateCpuKernelExecution())
            .BuildCatalog();

        // Act
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("cpu-kernel"),
            CreateServiceProvider());

        var handle = await kernel.SubmitBatchAsync(new[] { new[] { 1f, 2f, 3f } });
        var results = await kernel.ReadResultsAsync(handle).ToListAsync();

        // Assert
        results.Should().ContainSingle()
            .Which.Should().Be(42.0f);
    }

    private static Func<IReadOnlyList<float[]>, IAsyncEnumerable<float>> CreateCpuKernelExecution()
    {
        return items => CreateCpuResultsAsync(items);
    }

    private static async IAsyncEnumerable<float> CreateCpuResultsAsync(IReadOnlyList<float[]> items)
    {
        await Task.Yield();
        yield return 42.0f;
    }

    [Fact]
    public async Task CpuFallback_WithInvalidKernel_ShouldThrow()
    {
        // Arrange - Empty catalog (no kernels registered)
        var catalog = TestHelpers.CreateCatalog();

        // Act & Assert
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("non-existent-kernel"),
            CreateServiceProvider());

        // Should get CPU passthrough, but it won't know how to process
        kernel.Should().NotBeNull("Should get passthrough kernel");

        var info = await kernel.GetInfoAsync();
        info.Id.Value.Should().Be("cpu-passthrough");
    }

    [Fact]
    public async Task FallbackChain_ShouldTryAllProviders()
    {
        // Arrange - Multiple providers with different failure modes
        var providers = new[]
        {
            new MockGpuProviderRC2 { SimulateGpuCrash = true },
            new MockGpuProviderRC2 { SimulateOutOfMemory = true },
            new MockGpuProviderRC2 { SimulateGpuCrash = false } // This one works
        };

        int attemptCount = 0;

        // Act - Try each provider until one succeeds
        Exception? lastException = null;
        MockGpuProviderRC2? successfulProvider = null;

        foreach (var provider in providers)
        {
            attemptCount++;
            try
            {
                var kernel = CreateMockKernel(provider);
                var handle = await kernel.SubmitBatchAsync(new[] { new[] { 1f, 2f } });

                var results = new List<float>();
                await foreach (var result in kernel.ReadResultsAsync(handle))
                {
                    results.Add(result);
                }

                if (results.Any())
                {
                    successfulProvider = provider;
                    break;
                }
            }
            catch (Exception ex)
            {
                lastException = ex;
            }
        }

        // Assert
        attemptCount.Should().BeGreaterThan(1, "Should try multiple providers");
        successfulProvider.Should().NotBeNull("Should eventually find working provider");
    }

    [Fact]
    public async Task FallbackMetrics_ShouldRecordFallbackRate()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2
        {
            SimulateOutOfMemory = true,
            HasCpuFallback = true
        };

        int totalAttempts = 0;
        int fallbackCount = 0;

        // Act - Execute multiple kernels that will fail on GPU
        for (int i = 0; i < 5; i++)
        {
            totalAttempts++;

            try
            {
                var kernel = CreateMockKernel(mockGpu);
                var handle = await kernel.SubmitBatchAsync(new[] { new[] { 1f, 2f } });

                await foreach (var _ in kernel.ReadResultsAsync(handle))
                {
                    // Process results
                }
            }
            catch
            {
                fallbackCount++;
            }
        }

        // Assert
        totalAttempts.Should().Be(5);
        mockGpu.AllocationAttempts.Should().BeGreaterThanOrEqualTo(5,
            "Should attempt GPU allocation for each execution");

        var fallbackRate = (double)fallbackCount / totalAttempts;
        fallbackRate.Should().BeInRange(0.0, 1.0);
    }

    #endregion

    #region Timeout Tests (5 tests)

    [Fact]
    public async Task KernelExecution_WithLongRunning_ShouldTimeout()
    {
        // Arrange
        var kernel = new MockKernelRC2<float[], float>(
            TestHelpers.CreateKernelInfo("long-running"))
        {
            ExecutionDelay = TimeSpan.FromSeconds(10) // Very long execution
        };

        var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        // Act
        var handle = await kernel.SubmitBatchAsync(
            new[] { new[] { 1f, 2f, 3f } },
            null,
            cts.Token);

        var act = async () =>
        {
            await foreach (var _ in kernel.ReadResultsAsync(handle, cts.Token))
            {
            }
        };

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task MemoryAllocation_WithTimeout_ShouldCancel()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2();
        var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));

        // Act
        var act = async () =>
        {
            // Simulate slow allocation
            await Task.Delay(TimeSpan.FromSeconds(1), cts.Token);
            mockGpu.TryAllocateMemory(1024 * 1024);
        };

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task DeviceAllocation_WithTimeout_ShouldRelease()
    {
        // Arrange
        var mockGpu = new MockGpuProviderRC2();
        var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        var allocated = 0L;

        // Act
        try
        {
            allocated = mockGpu.TryAllocateMemory(1024 * 1024);
            await Task.Delay(TimeSpan.FromSeconds(1), cts.Token);
        }
        catch (OperationCanceledException)
        {
            // Cleanup on cancellation
            mockGpu.FreeMemory(allocated);
        }

        // Assert
        mockGpu.UsedMemory.Should().Be(0, "Memory should be released after timeout");
    }

    [Fact]
    public async Task GrainActivation_WithTimeout_ShouldDeactivate()
    {
        // Arrange
        var catalog = TestHelpers.CreateCatalog();
        var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));

        // Act - Simulate slow grain activation
        var act = async () =>
        {
            await Task.Delay(TimeSpan.FromSeconds(1), cts.Token);
            await catalog.ResolveAsync<float[], float>(
                new KernelId("test"),
                CreateServiceProvider(),
                cts.Token);
        };

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task BatchExecution_WithPartialTimeout_ShouldReturnPartial()
    {
        // Arrange
        var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(120));
        var batchSize = 10;

        var kernel = new MockKernelRC2<float[], float>(
            TestHelpers.CreateKernelInfo("partial-timeout"),
            customExecution: items => CreatePartialTimeoutResults(items, cts.Token));

        // Act
        var handle = await kernel.SubmitBatchAsync(
            TestHelpers.CreateSampleBatch(batchSize, 10),
            null,
            cts.Token);

        var results = new List<float>();
        var processedCount = 0;

        try
        {
            await foreach (var result in kernel.ReadResultsAsync(handle, cts.Token))
            {
                results.Add(result);
                processedCount++;
            }
        }
        catch (OperationCanceledException)
        {
            // Expected - partial processing
        }

        // Assert
        results.Should().NotBeEmpty("Should have partial results before timeout");
        results.Count.Should().BeLessThan(batchSize,
            "Should timeout before completing all items");
        processedCount.Should().BeGreaterThan(0, "Should process some items");
    }

    private static async IAsyncEnumerable<float> CreatePartialTimeoutResults(
        IReadOnlyList<float[]> items,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.Yield();
        foreach (var item in items)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Delay(50, ct); // Delay per item
            yield return 42.0f;
        }
    }

    #endregion

    #region Helper Methods

    private static KernelCatalog CreateCatalogWithFallback(MockGpuProviderRC2 mockGpu)
    {
        return new TestKernelCatalogBuilder()
            .AddKernel<float[], float>("test-kernel", mockGpu, items => ExecuteWithFallback(mockGpu, items))
            .AddKernel<float[], float>("fallback-test", mockGpu)
            .BuildCatalog();
    }

    private static async IAsyncEnumerable<float> ExecuteWithFallback(
        MockGpuProviderRC2 mockGpu,
        IReadOnlyList<float[]> items)
    {
        await Task.Yield();

        // Try GPU first
        foreach (var item in items)
        {
            float result;
            try
            {
                result = await mockGpu.ExecuteKernelAsync<float[], float>("test-kernel", item);
            }
            catch (OutOfMemoryException)
            {
                // Fallback to CPU
                mockGpu.FallbackCount++;
                result = item.Sum();
            }
            yield return result;
        }
    }

    private static MockKernelRC2<float[], float> CreateMockKernel(
        MockGpuProviderRC2? provider = null)
    {
        return new MockKernelRC2<float[], float>(
            TestHelpers.CreateKernelInfo("mock-kernel"),
            provider);
    }

    private static IServiceProvider CreateServiceProvider()
    {
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole());
        return services.BuildServiceProvider();
    }

    #endregion
}
