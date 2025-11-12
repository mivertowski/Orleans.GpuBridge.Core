using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime;

namespace Orleans.GpuBridge.Tests.RC2.TestingFramework;

/// <summary>
/// Helper utilities for RC2 error handling tests
/// </summary>
public static class TestHelpers
{
    /// <summary>
    /// Creates a KernelCatalog with mock GPU provider
    /// </summary>
    public static KernelCatalog CreateCatalog(
        MockGpuProviderRC2? mockProvider = null,
        bool hasCpuFallback = true,
        Action<KernelCatalogOptions>? configure = null)
    {
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Debug));

        var options = new KernelCatalogOptions();
        configure?.Invoke(options);

        services.Configure<KernelCatalogOptions>(opts =>
        {
            foreach (var descriptor in options.Descriptors)
            {
                opts.Descriptors.Add(descriptor);
            }
        });

        var serviceProvider = services.BuildServiceProvider();
        var logger = serviceProvider.GetRequiredService<ILogger<KernelCatalog>>();
        var catalogOptions = serviceProvider.GetRequiredService<IOptions<KernelCatalogOptions>>();

        return new KernelCatalog(logger, catalogOptions);
    }

    /// <summary>
    /// Creates a test kernel descriptor with CPU fallback
    /// </summary>
    public static KernelDescriptor CreateTestKernelDescriptor<TIn, TOut>(
        string kernelId,
        MockGpuProviderRC2? mockProvider = null,
        Func<TIn, Task<TOut>>? execution = null)
        where TIn : notnull
        where TOut : notnull
    {
        return new KernelDescriptor
        {
            Id = new KernelId(kernelId),
            InType = typeof(TIn),
            OutType = typeof(TOut),
            Factory = sp =>
            {
                var info = new KernelInfo(
                    new KernelId(kernelId),
                    $"Test kernel: {kernelId}",
                    typeof(TIn),
                    typeof(TOut),
                    SupportsGpu: true,
                    PreferredBatchSize: 64);

                return new MockKernelRC2<TIn, TOut>(info, mockProvider, execution);
            }
        };
    }

    /// <summary>
    /// Creates a kernel info for testing
    /// </summary>
    public static KernelInfo CreateKernelInfo(
        string id = "test-kernel",
        string description = "Test kernel",
        Type? inputType = null,
        Type? outputType = null,
        bool supportsGpu = true,
        int preferredBatchSize = 64)
    {
        return new KernelInfo(
            new KernelId(id),
            description,
            inputType ?? typeof(float[]),
            outputType ?? typeof(float),
            supportsGpu,
            preferredBatchSize);
    }

    /// <summary>
    /// Creates sample float array data for testing
    /// </summary>
    public static float[] CreateSampleData(int size = 100)
    {
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = i * 0.1f;
        }
        return data;
    }

    /// <summary>
    /// Creates batch of sample data
    /// </summary>
    public static IReadOnlyList<float[]> CreateSampleBatch(int batchSize = 10, int vectorSize = 100)
    {
        var batch = new List<float[]>(batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            batch.Add(CreateSampleData(vectorSize));
        }
        return batch;
    }

    /// <summary>
    /// Waits for a condition with timeout
    /// </summary>
    public static async Task<bool> WaitForConditionAsync(
        Func<bool> condition,
        TimeSpan timeout,
        TimeSpan pollInterval = default)
    {
        if (pollInterval == default)
            pollInterval = TimeSpan.FromMilliseconds(50);

        var endTime = DateTimeOffset.UtcNow + timeout;

        while (DateTimeOffset.UtcNow < endTime)
        {
            if (condition())
                return true;

            await Task.Delay(pollInterval);
        }

        return false;
    }

    /// <summary>
    /// Creates a cancellation token with timeout
    /// </summary>
    public static CancellationToken CreateTimeoutToken(TimeSpan timeout)
    {
        var cts = new CancellationTokenSource(timeout);
        return cts.Token;
    }
}

/// <summary>
/// Builder for creating kernel catalog options for tests
/// </summary>
public class TestKernelCatalogBuilder
{
    private readonly List<KernelDescriptor> _descriptors = new();

    public TestKernelCatalogBuilder AddKernel<TIn, TOut>(
        string kernelId,
        MockGpuProviderRC2? mockProvider = null,
        Func<TIn, Task<TOut>>? execution = null)
        where TIn : notnull
        where TOut : notnull
    {
        _descriptors.Add(TestHelpers.CreateTestKernelDescriptor<TIn, TOut>(kernelId, mockProvider, execution));
        return this;
    }

    public KernelCatalogOptions Build()
    {
        var options = new KernelCatalogOptions();
        foreach (var descriptor in _descriptors)
        {
            options.Descriptors.Add(descriptor);
        }
        return options;
    }

    public KernelCatalog BuildCatalog()
    {
        var options = Build();
        return TestHelpers.CreateCatalog(configure: opts =>
        {
            foreach (var descriptor in options.Descriptors)
            {
                opts.Descriptors.Add(descriptor);
            }
        });
    }
}
