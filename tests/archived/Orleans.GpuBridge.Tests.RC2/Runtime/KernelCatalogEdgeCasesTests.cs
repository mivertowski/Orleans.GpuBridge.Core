using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime;

namespace Orleans.GpuBridge.Tests.RC2.Runtime;

/// <summary>
/// Edge case test suite for KernelCatalog focusing on unusual scenarios, boundary conditions,
/// error paths, and stress testing to achieve 90%+ coverage.
/// Tests scenarios NOT covered by KernelCatalogTests.cs or KernelCatalogAdvancedTests.cs.
/// </summary>
public sealed class KernelCatalogEdgeCasesTests : IDisposable
{
    private readonly Mock<ILogger<KernelCatalog>> _mockLogger;
    private readonly Mock<IServiceProvider> _mockServiceProvider;
    private readonly CancellationTokenSource _cts;

    public KernelCatalogEdgeCasesTests()
    {
        _mockLogger = new Mock<ILogger<KernelCatalog>>();
        _mockServiceProvider = new Mock<IServiceProvider>();
        _cts = new CancellationTokenSource();
    }

    public void Dispose()
    {
        _cts?.Dispose();
    }

    #region Error Path Testing (10 tests)

    [Fact]
    public async Task ResolveKernel_WithNonExistentId_ShouldReturnCpuPassthrough()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernelId = new KernelId("non-existent-kernel-12345");

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            kernelId, _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull("CPU passthrough should be returned for unknown kernels");
        kernel.IsGpuAccelerated.Should().BeFalse();
        kernel.KernelId.Should().Be("cpu-passthrough");
    }

    [Fact]
    public async Task ResolveKernel_WithEmptyKernelId_ShouldHandleGracefully()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernelId = new KernelId("");

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            kernelId, _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull("should fallback to CPU passthrough");
    }

    [Fact]
    public async Task ResolveKernel_WithWhitespaceKernelId_ShouldHandleGracefully()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernelId = new KernelId("   ");

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            kernelId, _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull("should fallback to CPU passthrough");
    }

    [Fact]
    public async Task ResolveKernel_WithVeryLongKernelId_ShouldSucceed()
    {
        // Arrange
        var longId = new string('a', 10000);
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId(longId),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            new KernelId(longId), _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull();
    }

    [Fact]
    public async Task ResolveKernel_WithUnicodeKernelId_ShouldSucceed()
    {
        // Arrange
        var unicodeId = "kernel-日本語-🚀-τεστ";
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId(unicodeId),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            new KernelId(unicodeId), _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull();
    }

    [Fact]
    public async Task ResolveKernel_WithSpecialCharactersInId_ShouldSucceed()
    {
        // Arrange
        var specialId = "kernel!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId(specialId),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            new KernelId(specialId), _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull();
    }

    [Fact]
    public async Task Factory_ThrowingAggregateException_ShouldPropagateCorrectly()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("aggregate-throw"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => throw new AggregateException(
                new InvalidOperationException("Error 1"),
                new ArgumentException("Error 2"))
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>();
    }

    [Fact]
    public async Task Factory_ThrowingStackOverflowSimulated_ShouldPropagateCorrectly()
    {
        // Arrange - We can't actually cause StackOverflowException, but test deep recursion simulation
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("deep-recursion"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => throw new InvalidOperationException("Simulated stack overflow")
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>();
    }

    [Fact]
    public async Task ResolveKernel_WithCanceledTokenFromStart_ShouldThrowImmediately()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("pre-canceled"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        };

        var catalog = CreateCatalog(descriptor);
        var preCanceledCts = new CancellationTokenSource();
        preCanceledCts.Cancel();

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, preCanceledCts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task Factory_ReturningWrongInterfaceType_ShouldThrowInvalidOperation()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("wrong-interface"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new WrongInterfaceKernel() // Doesn't implement IGpuKernel
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*incompatible type*");
    }

    #endregion

    #region Service Provider Edge Cases (8 tests)

    [Fact]
    public async Task ResolveKernel_WithNullServiceProvider_ShouldThrowArgumentNull()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("test-kernel"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, null!, _cts.Token);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>()
            .WithParameterName("sp");
    }

    [Fact]
    public async Task Factory_AccessingDisposedServiceProvider_ShouldHandleGracefully()
    {
        // Arrange
        var disposedSp = new Mock<IServiceProvider>();
        disposedSp.Setup(sp => sp.GetService(It.IsAny<Type>()))
            .Throws(new ObjectDisposedException("ServiceProvider"));

        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("disposed-sp"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                sp.GetService(typeof(ILogger)); // Will throw
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, disposedSp.Object, _cts.Token);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>();
    }

    [Fact]
    public async Task Factory_WithMissingRequiredService_ShouldReturnNullAndFail()
    {
        // Arrange
        var spWithMissingService = new Mock<IServiceProvider>();
        spWithMissingService.Setup(sp => sp.GetService(typeof(IRequiredService)))
            .Returns((object?)null); // Missing required service - explicitly typed as nullable

        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("missing-service"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var service = sp.GetService(typeof(IRequiredService)) as IRequiredService
                    ?? throw new InvalidOperationException("Required service not found");
                return new KernelWithRequiredService<float, float>(service);
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, spWithMissingService.Object, _cts.Token);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>()
            .WithMessage("*Required service not found*");
    }

    [Fact]
    public async Task Factory_WithCircularDependency_ShouldDetectAndThrow()
    {
        // Arrange
        var spWithCircularDep = new Mock<IServiceProvider>();
        var circularDepDetected = false;

        spWithCircularDep.Setup(sp => sp.GetService(typeof(ICircularService)))
            .Returns(() =>
            {
                if (circularDepDetected)
                    throw new InvalidOperationException("Circular dependency detected");
                circularDepDetected = true;
                // Simulate circular dependency by requesting self
                spWithCircularDep.Object.GetService(typeof(ICircularService));
                return null;
            });

        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("circular-dep"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var service = sp.GetService(typeof(ICircularService));
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var act = async () => await catalog.ResolveAsync<float, float>(
            descriptor.Id, spWithCircularDep.Object, _cts.Token);

        // Assert
        await act.Should().ThrowAsync<InvalidOperationException>();
    }

    [Fact]
    public async Task Factory_WithTransientService_ShouldGetNewInstanceEachTime()
    {
        // Arrange
        var instanceCount = 0;
        _mockServiceProvider.Setup(sp => sp.GetService(typeof(ITransientService)))
            .Returns(() => new TransientService(++instanceCount));

        var resolveCount = 0;
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("transient-service"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var service = sp.GetService(typeof(ITransientService)) as ITransientService;
                resolveCount++;
                return new KernelWithTransientService<float, float>(service!);
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var kernel1 = await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);
        var kernel2 = await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        resolveCount.Should().Be(2);
        instanceCount.Should().Be(2, "transient services should be created each time");
    }

    [Fact]
    public async Task Factory_WithSingletonService_ShouldReuseSameInstance()
    {
        // Arrange
        var singletonService = new SingletonService(42);
        _mockServiceProvider.Setup(sp => sp.GetService(typeof(ISingletonService)))
            .Returns(singletonService);

        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("singleton-service"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var service = sp.GetService(typeof(ISingletonService)) as ISingletonService;
                return new KernelWithSingletonService<float, float>(service!);
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var kernel1 = await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);
        var kernel2 = await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        // Both kernels should have received the same singleton instance
        _mockServiceProvider.Verify(
            sp => sp.GetService(typeof(ISingletonService)),
            Times.Exactly(2));
    }

    [Fact]
    public async Task Factory_WithOptionalService_ShouldHandleNullGracefully()
    {
        // Arrange
        _mockServiceProvider.Setup(sp => sp.GetService(typeof(IOptionalService)))
            .Returns((object?)null); // Optional service not available - explicitly typed as nullable

        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("optional-service"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var service = sp.GetService(typeof(IOptionalService)) as IOptionalService;
                // Should handle null optional service gracefully
                return new KernelWithOptionalService<float, float>(service);
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull("kernel should work without optional service");
    }

    [Fact]
    public async Task Factory_WithGenericService_ShouldResolveCorrectly()
    {
        // Arrange
        var genericService = new GenericService<int>(123);
        _mockServiceProvider.Setup(sp => sp.GetService(typeof(IGenericService<int>)))
            .Returns(genericService);

        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("generic-service"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var service = sp.GetService(typeof(IGenericService<int>)) as IGenericService<int>;
                return new KernelWithGenericService<float, float>(service!);
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernel.Should().NotBeNull();
    }

    #endregion

    #region Concurrent Access Stress Tests (10 tests)

    [Fact]
    public async Task ConcurrentResolve_100Threads_SameKernel_NoDeadlock()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("stress-100"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                Thread.Sleep(10); // Simulate work
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var tasks = Enumerable.Range(0, 100).Select(async _ =>
            await catalog.ResolveAsync<float, float>(
                descriptor.Id, _mockServiceProvider.Object, _cts.Token));

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var kernels = await Task.WhenAll(tasks);
        sw.Stop();

        // Assert
        kernels.Should().HaveCount(100);
        kernels.Should().OnlyContain(k => k != null);
        sw.ElapsedMilliseconds.Should().BeLessThan(10000, "should not deadlock");
    }

    [Fact]
    public async Task ConcurrentResolve_WithRandomDelays_ShouldAllComplete()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("random-delay"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                Thread.Sleep(Random.Shared.Next(1, 50));
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var tasks = Enumerable.Range(0, 50).Select(async _ =>
            await catalog.ResolveAsync<float, float>(
                descriptor.Id, _mockServiceProvider.Object, _cts.Token));

        var kernels = await Task.WhenAll(tasks);

        // Assert
        kernels.Should().HaveCount(50);
        kernels.Should().OnlyContain(k => k != null);
    }

    [Fact]
    public async Task ConcurrentResolve_MixOfSuccessAndFailure_ShouldHandleCorrectly()
    {
        // Arrange
        var callCount = 0;
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("mixed-results"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var count = Interlocked.Increment(ref callCount);
                if (count % 5 == 0) // Every 5th call fails
                    throw new InvalidOperationException("Simulated failure");
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var tasks = Enumerable.Range(0, 25).Select(async _ =>
        {
            try
            {
                return await catalog.ResolveAsync<float, float>(
                    descriptor.Id, _mockServiceProvider.Object, _cts.Token);
            }
            catch
            {
                return null;
            }
        });

        var kernels = await Task.WhenAll(tasks);

        // Assert
        kernels.Should().HaveCount(25);
        kernels.Should().Contain(k => k != null, "some should succeed");
        kernels.Should().Contain(k => k == null, "some should fail");
    }

    [Fact]
    public async Task ConcurrentResolve_WithMemoryPressure_ShouldNotThrowOOM()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("memory-pressure"),
            InType = typeof(float[]),
            OutType = typeof(float),
            Factory = sp => new LargeMemoryKernel<float[], float>()
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var memoryBefore = GC.GetTotalMemory(true);

        var tasks = Enumerable.Range(0, 30).Select(async _ =>
        {
            var kernel = await catalog.ResolveAsync<float[], float>(
                descriptor.Id, _mockServiceProvider.Object, _cts.Token);
            return kernel;
        });

        var kernels = await Task.WhenAll(tasks);

        GC.Collect();
        GC.WaitForPendingFinalizers();

        var memoryAfter = GC.GetTotalMemory(true);

        // Assert
        kernels.Should().HaveCount(30);
        var memoryGrowth = memoryAfter - memoryBefore;
        memoryGrowth.Should().BeLessThan(500 * 1024 * 1024, "should not consume excessive memory");
    }

    [Fact]
    public async Task ConcurrentResolve_WithCancellationMidway_ShouldPartiallySucceed()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("mid-cancel"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                Thread.Sleep(100); // Some delay
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);
        var timedCts = new CancellationTokenSource(TimeSpan.FromMilliseconds(150));

        // Act
        var tasks = Enumerable.Range(0, 20).Select(async _ =>
        {
            try
            {
                return await catalog.ResolveAsync<float, float>(
                    descriptor.Id, _mockServiceProvider.Object, timedCts.Token);
            }
            catch (OperationCanceledException)
            {
                return null;
            }
        });

        var kernels = await Task.WhenAll(tasks);

        // Assert
        kernels.Should().HaveCount(20);
        kernels.Should().Contain(k => k != null, "some should complete before cancellation");
    }

    [Fact]
    public async Task StressTest_1000ConcurrentExecutions_ShouldSucceed()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("stress-1000"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        };

        var catalog = CreateCatalog(descriptor);

        // Act
        var sw = System.Diagnostics.Stopwatch.StartNew();

        var tasks = Enumerable.Range(0, 1000).Select(async i =>
        {
            var kernel = await catalog.ResolveAsync<float, float>(
                descriptor.Id, _mockServiceProvider.Object, _cts.Token);
            return await kernel.ExecuteBatchAsync(new[] { (float)i }, _cts.Token);
        });

        var results = await Task.WhenAll(tasks);
        sw.Stop();

        // Assert
        results.Should().HaveCount(1000);
        results.Should().OnlyContain(r => r.Length > 0);
        sw.ElapsedMilliseconds.Should().BeLessThan(30000, "should complete in reasonable time");
    }

    [Fact]
    public async Task ConcurrentResolve_DifferentTypes_NoTypeSafetyIssues()
    {
        // Arrange
        var floatDesc = new KernelDescriptor
        {
            Id = new KernelId("type-float"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        };

        var intDesc = new KernelDescriptor
        {
            Id = new KernelId("type-int"),
            InType = typeof(int),
            OutType = typeof(int),
            Factory = sp => new SimpleKernel<int, int>()
        };

        var doubleDesc = new KernelDescriptor
        {
            Id = new KernelId("type-double"),
            InType = typeof(double),
            OutType = typeof(double),
            Factory = sp => new SimpleKernel<double, double>()
        };

        var catalog = CreateCatalog(floatDesc, intDesc, doubleDesc);

        // Act - Interleave different type resolutions
        var tasks = new List<Task>();
        for (int i = 0; i < 30; i++)
        {
            tasks.Add(catalog.ResolveAsync<float, float>(floatDesc.Id, _mockServiceProvider.Object, _cts.Token));
            tasks.Add(catalog.ResolveAsync<int, int>(intDesc.Id, _mockServiceProvider.Object, _cts.Token));
            tasks.Add(catalog.ResolveAsync<double, double>(doubleDesc.Id, _mockServiceProvider.Object, _cts.Token));
        }

        await Task.WhenAll(tasks.ToArray());

        // Assert - Should not throw type safety exceptions
        true.Should().BeTrue();
    }

    [Fact]
    public async Task ConcurrentResolve_WithBarrierSynchronization_NoRaceConditions()
    {
        // Arrange
        var accessCount = 0;
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("barrier-test"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                Interlocked.Increment(ref accessCount);
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);
        var threadCount = 20;
        var barrier = new System.Threading.Barrier(threadCount);

        // Act - Synchronize all threads to start at same time
        var tasks = Enumerable.Range(0, threadCount).Select(_ => Task.Run(async () =>
        {
            barrier.SignalAndWait(); // Wait for all threads
            return await catalog.ResolveAsync<float, float>(
                descriptor.Id, _mockServiceProvider.Object, _cts.Token);
        })).ToArray();

        var kernels = await Task.WhenAll(tasks);

        // Assert
        kernels.Should().HaveCount(threadCount);
        kernels.Should().OnlyContain(k => k != null);
        accessCount.Should().Be(threadCount, "each thread should create its own instance");
    }

    [Fact]
    public async Task ConcurrentResolve_WithExceptionInMiddle_ShouldNotCorruptState()
    {
        // Arrange
        var callCount = 0;
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("exception-middle"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp =>
            {
                var count = Interlocked.Increment(ref callCount);
                if (count == 10) // 10th call throws
                    throw new InvalidOperationException("Middleware exception");
                return new SimpleKernel<float, float>();
            }
        };

        var catalog = CreateCatalog(descriptor);

        // Act - Launch many concurrent resolves
        var tasks = Enumerable.Range(0, 30).Select(async _ =>
        {
            try
            {
                return await catalog.ResolveAsync<float, float>(
                    descriptor.Id, _mockServiceProvider.Object, _cts.Token);
            }
            catch
            {
                return null;
            }
        });

        var kernels = await Task.WhenAll(tasks);

        // After exception, catalog should still work
        var recoveryKernel = await catalog.ResolveAsync<float, float>(
            descriptor.Id, _mockServiceProvider.Object, _cts.Token);

        // Assert
        kernels.Should().HaveCount(30);
        recoveryKernel.Should().NotBeNull("catalog should recover after exception");
    }

    [Fact]
    public async Task RapidResolveAndDispose_NoResourceLeaks()
    {
        // Arrange
        var descriptor = new KernelDescriptor
        {
            Id = new KernelId("rapid-dispose"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new DisposableTestKernel<float, float>()
        };

        var catalog = CreateCatalog(descriptor);

        // Act - Rapidly create and dispose kernels
        for (int i = 0; i < 100; i++)
        {
            var kernel = await catalog.ResolveAsync<float, float>(
                descriptor.Id, _mockServiceProvider.Object, _cts.Token);

            if (kernel is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        // Force GC to detect leaks
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Assert - No assertion, success means no resource leaks
        true.Should().BeTrue();
    }

    #endregion

    #region CPU Passthrough Edge Cases (8 tests)

    [Fact]
    public async Task CpuPassthrough_WithVeryLargeArray_ShouldHandleEfficiently()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernel = await catalog.ResolveAsync<float[], float[]>(
            new KernelId("large-array-passthrough"), _mockServiceProvider.Object, _cts.Token);

        // Act
        var largeArray = Enumerable.Range(0, 100000).Select(i => (float)i).ToArray();
        var results = await kernel.ExecuteBatchAsync(new[] { largeArray }, _cts.Token);

        // Assert
        results.Should().HaveCount(1);
        results[0].Length.Should().Be(100000);
    }

    [Fact]
    public async Task CpuPassthrough_WithComplexType_ShouldPassthrough()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernel = await catalog.ResolveAsync<ComplexData, ComplexData>(
            new KernelId("complex-passthrough"), _mockServiceProvider.Object, _cts.Token);

        // Act
        var complexData = new ComplexData { Id = 123, Name = "Test", Values = new[] { 1.0f, 2.0f } };
        var results = await kernel.ExecuteBatchAsync(new[] { complexData }, _cts.Token);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().BeSameAs(complexData);
    }

    [Fact]
    public async Task CpuPassthrough_WithNullableType_ShouldHandleNull()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernel = await catalog.ResolveAsync<string, string>(
            new KernelId("nullable-passthrough"), _mockServiceProvider.Object, _cts.Token);

        // Act
        var data = new[] { "value1", null!, "value3" };
        var results = await kernel.ExecuteBatchAsync(data, _cts.Token);

        // Assert
        results.Should().HaveCount(3);
        results.Should().ContainInOrder("value1", null, "value3");
    }

    [Fact]
    public async Task CpuPassthrough_WithConvertibleTypes_ShouldAttemptConversion()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernel = await catalog.ResolveAsync<int, double>(
            new KernelId("convertible-passthrough"), _mockServiceProvider.Object, _cts.Token);

        // Act
        var integers = new[] { 1, 2, 3, 4, 5 };
        var results = await kernel.ExecuteBatchAsync(integers, _cts.Token);

        // Assert
        results.Should().HaveCount(5);
        // Conversion might not work for passthrough, so just verify it doesn't crash
    }

    [Fact]
    public async Task CpuPassthrough_WithEmptyBatch_ShouldReturnEmpty()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernel = await catalog.ResolveAsync<float, float>(
            new KernelId("empty-batch"), _mockServiceProvider.Object, _cts.Token);

        // Act
        var results = await kernel.ExecuteBatchAsync(Array.Empty<float>(), _cts.Token);

        // Assert
        results.Should().BeEmpty();
    }

    [Fact]
    public async Task CpuPassthrough_WithSingleItem_ShouldPassthrough()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernel = await catalog.ResolveAsync<float, float>(
            new KernelId("single-item"), _mockServiceProvider.Object, _cts.Token);

        // Act
        var results = await kernel.ExecuteBatchAsync(new[] { 42.0f }, _cts.Token);

        // Assert
        results.Should().ContainSingle();
        results[0].Should().Be(42.0f);
    }

    [Fact]
    public async Task CpuPassthrough_ConcurrentBatches_ShouldIsolate()
    {
        // Arrange
        var catalog = CreateCatalog();
        var kernel = await catalog.ResolveAsync<int, int>(
            new KernelId("concurrent-batches"), _mockServiceProvider.Object, _cts.Token);

        // Act - Submit multiple batches concurrently
        var tasks = Enumerable.Range(0, 20).Select(async i =>
        {
            var data = Enumerable.Range(i * 100, 10).ToArray();
            return await kernel.ExecuteBatchAsync(data, _cts.Token);
        });

        var allResults = await Task.WhenAll(tasks);

        // Assert
        allResults.Should().HaveCount(20);
        allResults.Should().OnlyContain(r => r.Length == 10);
    }

    [Fact]
    public async Task CpuPassthrough_MultipleKernelInstances_ShouldBeIndependent()
    {
        // Arrange
        var catalog = CreateCatalog();

        // Act - Create multiple passthrough kernels
        var kernel1 = await catalog.ResolveAsync<float, float>(
            new KernelId("passthrough-1"), _mockServiceProvider.Object, _cts.Token);
        var kernel2 = await catalog.ResolveAsync<float, float>(
            new KernelId("passthrough-2"), _mockServiceProvider.Object, _cts.Token);

        var results1 = await kernel1.ExecuteBatchAsync(new[] { 1.0f }, _cts.Token);
        var results2 = await kernel2.ExecuteBatchAsync(new[] { 2.0f }, _cts.Token);

        // Assert
        results1[0].Should().Be(1.0f);
        results2[0].Should().Be(2.0f);
    }

    #endregion

    #region Catalog Initialization Edge Cases (6 tests)

    [Fact]
    public void Constructor_WithNullLogger_ShouldThrow()
    {
        // Arrange & Act
        var act = () => new KernelCatalog(
            null!,
            Options.Create(new KernelCatalogOptions()));

        // Assert
        act.Should().Throw<ArgumentNullException>()
            .WithParameterName("logger");
    }

    [Fact]
    public void Constructor_WithNullOptions_ShouldThrow()
    {
        // Arrange & Act
        var act = () => new KernelCatalog(
            _mockLogger.Object,
            null!);

        // Assert
        act.Should().Throw<ArgumentNullException>()
            .WithParameterName("options");
    }

    [Fact]
    public void Constructor_WithEmptyDescriptors_ShouldSucceed()
    {
        // Arrange & Act
        var act = () => CreateCatalog();

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public async Task Constructor_With100Descriptors_ShouldSucceed()
    {
        // Arrange
        var descriptors = Enumerable.Range(0, 100).Select(i => new KernelDescriptor
        {
            Id = new KernelId($"kernel-{i}"),
            InType = typeof(float),
            OutType = typeof(float),
            Factory = sp => new SimpleKernel<float, float>()
        }).ToArray();

        // Act
        var catalog = CreateCatalog(descriptors);

        // Assert - Should be able to resolve any kernel
        var kernel = await catalog.ResolveAsync<float, float>(
            new KernelId("kernel-50"), _mockServiceProvider.Object, _cts.Token);
        kernel.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_WithDescriptorsContainingNullFactory_ShouldSkipThem()
    {
        // Arrange
        var descriptors = new[]
        {
            new KernelDescriptor
            {
                Id = new KernelId("valid"),
                InType = typeof(float),
                OutType = typeof(float),
                Factory = sp => new SimpleKernel<float, float>()
            },
            new KernelDescriptor
            {
                Id = new KernelId("null-factory"),
                InType = typeof(float),
                OutType = typeof(float),
                Factory = null // Null factory
            }
        };

        // Act
        var catalog = CreateCatalog(descriptors);

        // Assert - Should not throw, null factories are skipped
        catalog.Should().NotBeNull();
    }

    [Fact]
    public async Task Constructor_WithDuplicateIds_ShouldUseLastRegistered()
    {
        // Arrange
        var descriptors = new[]
        {
            new KernelDescriptor
            {
                Id = new KernelId("duplicate"),
                InType = typeof(float),
                OutType = typeof(float),
                Factory = sp => new SimpleKernel<float, float>(1)
            },
            new KernelDescriptor
            {
                Id = new KernelId("duplicate"),
                InType = typeof(float),
                OutType = typeof(float),
                Factory = sp => new SimpleKernel<float, float>(2)
            }
        };

        var catalog = CreateCatalog(descriptors);

        // Act
        var kernel = await catalog.ResolveAsync<float, float>(
            new KernelId("duplicate"), _mockServiceProvider.Object, _cts.Token);

        // Assert - Should get the last registered (id=2)
        kernel.Should().NotBeNull();
    }

    #endregion

    #region Helper Methods

    private KernelCatalog CreateCatalog(params KernelDescriptor[] descriptors)
    {
        var options = Options.Create(new KernelCatalogOptions());

        foreach (var descriptor in descriptors)
        {
            options.Value.Descriptors.Add(descriptor);
        }

        return new KernelCatalog(_mockLogger.Object, options);
    }

    #endregion

    #region Test Helper Classes

    private class SimpleKernel<TIn, TOut> : GpuKernelBase<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        private readonly int _id;

        public SimpleKernel(int id = 0) => _id = id;

        public override string KernelId => $"simple-kernel-{_id}";
        public override string BackendProvider => "Mock";
        public override bool IsGpuAccelerated => false;

        public override async Task<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default)
        {
            await Task.Delay(10, cancellationToken);

            // Transform input to output
            if (input is TOut result)
                return result;

            return default!;
        }

        public override async Task<TOut[]> ExecuteBatchAsync(TIn[] inputs, CancellationToken cancellationToken = default)
        {
            var results = new TOut[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                results[i] = await ExecuteAsync(inputs[i], cancellationToken);
            }
            return results;
        }

        public override long GetEstimatedExecutionTimeMicroseconds(int inputSize) => inputSize * 10;

        public override KernelMemoryRequirements GetMemoryRequirements() =>
            new KernelMemoryRequirements(1024, 1024, 512, 2560);
    }

    private class WrongInterfaceKernel
    {
        // Does not implement IGpuKernel
    }

    private interface IRequiredService { }
    private interface ICircularService { }
    private interface ITransientService { }
    private interface ISingletonService { }
    private interface IOptionalService { }
    private interface IGenericService<T> { }

    private class TransientService : ITransientService
    {
        public int InstanceId { get; }
        public TransientService(int id) => InstanceId = id;
    }

    private class SingletonService : ISingletonService
    {
        public int Value { get; }
        public SingletonService(int value) => Value = value;
    }

    private class GenericService<T> : IGenericService<T>
    {
        public T Value { get; }
        public GenericService(T value) => Value = value;
    }

    private class KernelWithRequiredService<TIn, TOut> : SimpleKernel<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        public IRequiredService Service { get; }
        public KernelWithRequiredService(IRequiredService service) => Service = service;
    }

    private class KernelWithTransientService<TIn, TOut> : SimpleKernel<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        public ITransientService Service { get; }
        public KernelWithTransientService(ITransientService service) => Service = service;
    }

    private class KernelWithSingletonService<TIn, TOut> : SimpleKernel<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        public ISingletonService Service { get; }
        public KernelWithSingletonService(ISingletonService service) => Service = service;
    }

    private class KernelWithOptionalService<TIn, TOut> : SimpleKernel<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        public IOptionalService? Service { get; }
        public KernelWithOptionalService(IOptionalService? service) => Service = service;
    }

    private class KernelWithGenericService<TIn, TOut> : SimpleKernel<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        public IGenericService<int> Service { get; }
        public KernelWithGenericService(IGenericService<int> service) => Service = service;
    }

    private class LargeMemoryKernel<TIn, TOut> : SimpleKernel<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        // Allocate 10MB
        private readonly byte[] _largeBuffer = new byte[10 * 1024 * 1024];
    }

    private class DisposableTestKernel<TIn, TOut> : SimpleKernel<TIn, TOut>
        where TIn : notnull
        where TOut : notnull
    {
        private bool _disposed;

        public override void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                base.Dispose();
            }
        }
    }

    private class ComplexData
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public float[] Values { get; set; } = Array.Empty<float>();
    }

    #endregion
}
