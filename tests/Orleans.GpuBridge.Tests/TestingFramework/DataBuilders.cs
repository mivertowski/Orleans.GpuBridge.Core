using System;
using System.Collections.Generic;
using System.Linq;
using Bogus;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Fluent test data builders using Builder pattern
/// </summary>
public static class TestDataBuilders
{
    public static KernelInfoBuilder KernelInfo() => new();
    public static ComputeDeviceBuilder ComputeDevice() => new();
    public static GpuExecutionHintsBuilder ExecutionHints() => new();
    public static TestKernelBuilder<TIn, TOut> Kernel<TIn, TOut>() where TIn : notnull where TOut : notnull => new();
    public static TestDataBuilder<T> Data<T>() where T : class => new();
}

/// <summary>
/// Builder for KernelInfo test data
/// </summary>
public class KernelInfoBuilder
{
    private KernelId _id = new("test-kernel");
    private string _displayName = "Test Kernel";
    private Type _inputType = typeof(float[]);
    private Type _outputType = typeof(float);
    private bool _supportsGpu = false;
    private int _preferredBatchSize = 1024;

    public KernelInfoBuilder WithId(string id) 
    {
        _id = new KernelId(id);
        return this;
    }

    public KernelInfoBuilder WithDisplayName(string displayName)
    {
        _displayName = displayName;
        return this;
    }

    public KernelInfoBuilder WithInputType<T>()
    {
        _inputType = typeof(T);
        return this;
    }

    public KernelInfoBuilder WithOutputType<T>()
    {
        _outputType = typeof(T);
        return this;
    }

    public KernelInfoBuilder WithGpuSupport(bool supportsGpu = true)
    {
        _supportsGpu = supportsGpu;
        return this;
    }

    public KernelInfoBuilder WithBatchSize(int batchSize)
    {
        _preferredBatchSize = batchSize;
        return this;
    }

    public KernelInfo Build() => new(_id, _displayName, _inputType, _outputType, _supportsGpu, _preferredBatchSize);
}

/// <summary>
/// Builder for ComputeDevice test data
/// </summary>
public class ComputeDeviceBuilder
{
    private int _index = 0;
    private string _name = "Test Device";
    private DeviceType _type = DeviceType.GPU;
    private long _totalMemoryBytes = 1024L * 1024 * 1024; // 1GB
    private int _computeUnits = 16;
    private string[] _capabilities = new[] { "CUDA", "OpenCL" };

    public ComputeDeviceBuilder WithIndex(int index)
    {
        _index = index;
        return this;
    }

    public ComputeDeviceBuilder WithName(string name)
    {
        _name = name;
        return this;
    }

    public ComputeDeviceBuilder WithType(DeviceType type)
    {
        _type = type;
        return this;
    }

    public ComputeDeviceBuilder WithMemory(long bytes)
    {
        _totalMemoryBytes = bytes;
        return this;
    }

    public ComputeDeviceBuilder WithComputeUnits(int units)
    {
        _computeUnits = units;
        return this;
    }

    public ComputeDeviceBuilder WithCapabilities(params string[] capabilities)
    {
        _capabilities = capabilities;
        return this;
    }

    public ComputeDeviceBuilder AsCpuDevice()
    {
        _type = DeviceType.CPU;
        _capabilities = new[] { "CPU", "AVX", "SSE" };
        return this;
    }

    public ComputeDeviceBuilder AsGpuDevice()
    {
        _type = DeviceType.GPU;
        _capabilities = new[] { "CUDA", "OpenCL", "Vulkan" };
        return this;
    }

    public IComputeDevice Build()
    {
        var device = new TestComputeDevice
        {
            DeviceId = Guid.NewGuid().ToString(),
            Index = _index,
            Name = _name,
            Type = _type,
            Vendor = "Test Vendor",
            Architecture = "Test Architecture",
            ComputeCapability = new Version(7, 5),
            TotalMemoryBytes = _totalMemoryBytes,
            AvailableMemoryBytes = _totalMemoryBytes,
            ComputeUnits = _computeUnits,
            MaxClockFrequencyMHz = 1500,
            MaxThreadsPerBlock = 1024,
            MaxWorkGroupDimensions = new[] { 1024, 1024, 64 },
            WarpSize = 32,
            Properties = new Dictionary<string, object> { ["Capabilities"] = _capabilities }
        };
        return device;
    }
}

/// <summary>
/// Builder for GpuExecutionHints test data
/// </summary>
public class GpuExecutionHintsBuilder
{
    private int? _preferredBatchSize;
    private bool _preferGpu = true;
    private int _timeoutMs = 30000;

    public GpuExecutionHintsBuilder WithBatchSize(int batchSize)
    {
        _preferredBatchSize = batchSize;
        return this;
    }

    public GpuExecutionHintsBuilder PreferGpu(bool prefer = true)
    {
        _preferGpu = prefer;
        return this;
    }

    public GpuExecutionHintsBuilder WithTimeout(int timeoutMs)
    {
        _timeoutMs = timeoutMs;
        return this;
    }

    public GpuExecutionHints Build() => new(
        PreferredDevice: null,
        HighPriority: false,
        MaxMicroBatch: _preferredBatchSize,
        Persistent: true,
        PreferGpu: _preferGpu,
        Timeout: _timeoutMs > 0 ? TimeSpan.FromMilliseconds(_timeoutMs) : null,
        MaxRetries: null);
}

/// <summary>
/// Builder for test kernel implementations
/// </summary>
public class TestKernelBuilder<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private KernelInfo _info;
    private Func<IReadOnlyList<TIn>, IAsyncEnumerable<TOut>>? _execution;
    private TimeSpan _delay = TimeSpan.Zero;
    private Exception? _exceptionToThrow;

    public TestKernelBuilder()
    {
        _info = TestDataBuilders.KernelInfo()
            .WithInputType<TIn>()
            .WithOutputType<TOut>()
            .Build();
    }

    public TestKernelBuilder<TIn, TOut> WithInfo(KernelInfo info)
    {
        _info = info;
        return this;
    }

    public TestKernelBuilder<TIn, TOut> WithExecution(Func<IReadOnlyList<TIn>, IAsyncEnumerable<TOut>> execution)
    {
        _execution = execution;
        return this;
    }

    public TestKernelBuilder<TIn, TOut> WithDelay(TimeSpan delay)
    {
        _delay = delay;
        return this;
    }

    public TestKernelBuilder<TIn, TOut> ThrowsException(Exception exception)
    {
        _exceptionToThrow = exception;
        return this;
    }

    public TestKernel<TIn, TOut> Build() => new(_info, _execution, _delay, _exceptionToThrow);
}

/// <summary>
/// Generic test data builder using Bogus
/// </summary>
public class TestDataBuilder<T> where T : class
{
    private int _count = 1;
    private Faker<T>? _faker;

    public TestDataBuilder<T> WithCount(int count)
    {
        _count = count;
        return this;
    }

    public TestDataBuilder<T> WithFaker(Faker<T> faker)
    {
        _faker = faker;
        return this;
    }

    public TestDataBuilder<T> WithFaker(Action<Faker<T>> configure)
    {
        _faker = new Faker<T>();
        configure(_faker);
        return this;
    }

    public T Build()
    {
        if (_faker != null)
        {
            return _faker.Generate();
        }

        // Default generation for common types
        if (typeof(T) == typeof(float[]))
        {
            var faker = new Faker();
            return (T)(object)faker.Make(_count, () => faker.Random.Float());
        }

        if (typeof(T) == typeof(int[]))
        {
            var faker = new Faker();
            return (T)(object)faker.Make(_count, () => faker.Random.Int());
        }

        throw new InvalidOperationException($"No default faker configured for type {typeof(T)}");
    }

    public List<T> BuildMany() => BuildMany(_count);

    public List<T> BuildMany(int count)
    {
        if (_faker != null)
        {
            return _faker.Generate(count);
        }

        // For arrays, just build one array
        if (typeof(T).IsArray)
        {
            return new List<T> { Build() };
        }

        throw new InvalidOperationException($"No faker configured for generating multiple {typeof(T)}");
    }
}

/// <summary>
/// Faker extensions for GPU-specific data
/// </summary>
public static class FakerExtensions
{
    public static Faker<float[]> FloatArray(this Faker faker, int minSize = 1, int maxSize = 1000)
    {
        return new Faker<float[]>()
            .CustomInstantiator(f => f.Make(f.Random.Int(minSize, maxSize), () => f.Random.Float()));
    }

    public static Faker<int[]> IntArray(this Faker faker, int minSize = 1, int maxSize = 1000)
    {
        return new Faker<int[]>()
            .CustomInstantiator(f => f.Make(f.Random.Int(minSize, maxSize), () => f.Random.Int()));
    }

    public static KernelId GenerateKernelId(this Faker faker)
    {
        return new KernelId($"kernel/{faker.Random.AlphaNumeric(8)}");
    }

    public static Faker<GpuExecutionHints> ExecutionHints(this Faker faker)
    {
        return new Faker<GpuExecutionHints>()
            .CustomInstantiator(f => new GpuExecutionHints(
                PreferredDevice: f.Random.Int(0, 4),
                HighPriority: f.Random.Bool(),
                MaxMicroBatch: f.Random.Int(1, 4096),
                Persistent: f.Random.Bool(),
                PreferGpu: f.Random.Bool(),
                Timeout: TimeSpan.FromMilliseconds(f.Random.Int(1000, 60000)),
                MaxRetries: f.Random.Int(1, 5)
            ));
    }
}