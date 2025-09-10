using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Test implementation of IGpuKernel for testing purposes
/// </summary>
public class TestKernel<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly KernelInfo _info;
    private readonly Func<IReadOnlyList<TIn>, IAsyncEnumerable<TOut>>? _execution;
    private readonly TimeSpan _delay;
    private readonly Exception? _exceptionToThrow;
    private readonly Dictionary<string, KernelHandle> _activeHandles = new();

    public TestKernel(
        KernelInfo info,
        Func<IReadOnlyList<TIn>, IAsyncEnumerable<TOut>>? execution = null,
        TimeSpan delay = default,
        Exception? exceptionToThrow = null)
    {
        _info = info;
        _execution = execution;
        _delay = delay;
        _exceptionToThrow = exceptionToThrow;
    }

    public async ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        if (_exceptionToThrow != null)
            throw _exceptionToThrow;

        if (_delay > TimeSpan.Zero)
            await Task.Delay(_delay, ct);

        var handle = KernelHandle.Create();
        _activeHandles[handle.Id] = handle;
        
        return handle;
    }

    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (_exceptionToThrow != null)
            throw _exceptionToThrow;

        if (!_activeHandles.TryGetValue(handle.Id, out var storedHandle) || !handle.Equals(storedHandle))
            throw new ArgumentException("Invalid or unknown handle", nameof(handle));

        try
        {
            if (_execution != null)
            {
                // Use custom execution logic
                await foreach (var result in _execution(Array.Empty<TIn>()).WithCancellation(ct))
                {
                    yield return result;
                }
            }
            else
            {
                // Default behavior - return a single default value
                if (typeof(TOut) == typeof(float))
                {
                    yield return (TOut)(object)42.0f;
                }
                else if (typeof(TOut) == typeof(int))
                {
                    yield return (TOut)(object)42;
                }
                else if (typeof(TOut) == typeof(float[]))
                {
                    yield return (TOut)(object)new[] { 1.0f, 2.0f, 3.0f };
                }
                else if (typeof(TOut) == typeof(int[]))
                {
                    yield return (TOut)(object)new[] { 1, 2, 3 };
                }
                else
                {
                    yield return default(TOut)!;
                }
            }
        }
        finally
        {
            _activeHandles.Remove(handle.Id);
        }
    }

    public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        return new(_info);
    }

    public bool IsHandleActive(KernelHandle handle) => _activeHandles.ContainsKey(handle.Id);
    public int ActiveHandleCount => _activeHandles.Count;
}

/// <summary>
/// Factory for creating test kernels with common configurations
/// </summary>
public static class TestKernelFactory
{
    public static TestKernel<float[], float> CreateVectorAddKernel()
    {
        var info = TestDataBuilders.KernelInfo()
            .WithId("test/vector-add")
            .WithDisplayName("Test Vector Add")
            .WithInputType<float[]>()
            .WithOutputType<float>()
            .WithGpuSupport(true)
            .Build();

        return new TestKernel<float[], float>(
            info,
            inputs => ProcessVectorAddAsync(inputs));
    }

    public static TestKernel<float[], float[]> CreateVectorMultiplyKernel()
    {
        var info = TestDataBuilders.KernelInfo()
            .WithId("test/vector-multiply")
            .WithDisplayName("Test Vector Multiply")
            .WithInputType<float[]>()
            .WithOutputType<float[]>()
            .WithGpuSupport(true)
            .Build();

        return new TestKernel<float[], float[]>(
            info,
            inputs => ProcessVectorMultiplyAsync(inputs));
    }

    public static TestKernel<TIn, TOut> CreateSlowKernel<TIn, TOut>(TimeSpan delay)
        where TIn : notnull
        where TOut : notnull
    {
        var info = TestDataBuilders.KernelInfo()
            .WithId("test/slow-kernel")
            .WithDisplayName("Test Slow Kernel")
            .WithInputType<TIn>()
            .WithOutputType<TOut>()
            .Build();

        return new TestKernel<TIn, TOut>(info, delay: delay);
    }

    public static TestKernel<TIn, TOut> CreateFailingKernel<TIn, TOut>(Exception exception)
        where TIn : notnull
        where TOut : notnull
    {
        var info = TestDataBuilders.KernelInfo()
            .WithId("test/failing-kernel")
            .WithDisplayName("Test Failing Kernel")
            .WithInputType<TIn>()
            .WithOutputType<TOut>()
            .Build();

        return new TestKernel<TIn, TOut>(info, exceptionToThrow: exception);
    }

    private static async IAsyncEnumerable<float> ProcessVectorAddAsync(IReadOnlyList<float[]> inputs)
    {
        await Task.Yield();
        
        foreach (var vector in inputs)
        {
            var sum = 0f;
            foreach (var value in vector)
            {
                sum += value;
            }
            yield return sum;
        }
    }

    private static async IAsyncEnumerable<float[]> ProcessVectorMultiplyAsync(IReadOnlyList<float[]> inputs)
    {
        await Task.Yield();
        
        foreach (var vector in inputs)
        {
            var result = new float[vector.Count];
            for (int i = 0; i < vector.Count; i++)
            {
                result[i] = vector[i] * 2.0f;
            }
            yield return result;
        }
    }
}

/// <summary>
/// Kernel that can be configured to simulate various GPU behaviors
/// </summary>
public class ConfigurableTestKernel<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly KernelInfo _info;
    private readonly TestKernelConfiguration _config;
    private readonly Dictionary<string, KernelHandle> _activeHandles = new();

    public ConfigurableTestKernel(KernelInfo info, TestKernelConfiguration? config = null)
    {
        _info = info;
        _config = config ?? new TestKernelConfiguration();
    }

    public async ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        // Simulate submission delay
        if (_config.SubmissionDelay > TimeSpan.Zero)
            await Task.Delay(_config.SubmissionDelay, ct);

        // Simulate submission failure
        if (_config.SubmissionFailureRate > 0 && Random.Shared.NextDouble() < _config.SubmissionFailureRate)
            throw new InvalidOperationException("Simulated submission failure");

        var handle = KernelHandle.Create();
        _activeHandles[handle.Id] = handle;
        
        return handle;
    }

    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (!_activeHandles.TryGetValue(handle.Id, out var storedHandle) || !handle.Equals(storedHandle))
            throw new ArgumentException("Invalid or unknown handle", nameof(handle));

        try
        {
            // Simulate execution delay
            if (_config.ExecutionDelay > TimeSpan.Zero)
                await Task.Delay(_config.ExecutionDelay, ct);

            // Simulate execution failure
            if (_config.ExecutionFailureRate > 0 && Random.Shared.NextDouble() < _config.ExecutionFailureRate)
                throw new InvalidOperationException("Simulated execution failure");

            // Return configured results
            for (int i = 0; i < _config.ResultCount; i++)
            {
                if (_config.ResultDelay > TimeSpan.Zero)
                    await Task.Delay(_config.ResultDelay, ct);

                yield return _config.ResultFactory != null 
                    ? (TOut)_config.ResultFactory() 
                    : default(TOut)!;
            }
        }
        finally
        {
            _activeHandles.Remove(handle.Id);
        }
    }

    public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        return new(_info);
    }

    public int ActiveHandleCount => _activeHandles.Count;
}

/// <summary>
/// Configuration for configurable test kernel behavior
/// </summary>
public class TestKernelConfiguration
{
    public TimeSpan SubmissionDelay { get; set; } = TimeSpan.Zero;
    public TimeSpan ExecutionDelay { get; set; } = TimeSpan.Zero;
    public TimeSpan ResultDelay { get; set; } = TimeSpan.Zero;
    public double SubmissionFailureRate { get; set; } = 0.0;
    public double ExecutionFailureRate { get; set; } = 0.0;
    public int ResultCount { get; set; } = 1;
    public Func<object>? ResultFactory { get; set; }
}