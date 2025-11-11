using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Orleans.GpuBridge.Tests.RC2.TestingFramework;
using Xunit;

namespace Orleans.GpuBridge.Tests.RC2.BridgeFX;

/// <summary>
/// Comprehensive edge case tests for pipeline operations.
/// Tests empty inputs, null handling, boundary conditions, exception scenarios, and memory pressure.
/// Target: 20+ tests covering all edge cases and error scenarios.
/// </summary>
public sealed class PipelineEdgeCasesTests : IDisposable
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IGpuBridge _bridge;
    private readonly ILogger<GpuPipeline> _logger;
    private readonly CancellationTokenSource _cts;

    public PipelineEdgeCasesTests()
    {
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Debug));
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        _serviceProvider = services.BuildServiceProvider();

        _bridge = _serviceProvider.GetRequiredService<IGpuBridge>();
        _logger = _serviceProvider.GetRequiredService<ILogger<GpuPipeline>>();
        _cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    }

    #region Empty and Null Input Tests (6 tests)

    [Fact]
    public async Task EmptyInput_ShouldReturnEmpty()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        var inputs = AsyncEnumerable.Empty<int>();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().BeEmpty();
    }

    [Fact]
    public async Task EmptyAfterFilter_ShouldReturnEmpty()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x > 1000)
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().BeEmpty();
    }

    [Fact]
    public async Task SingleItem_ShouldProcess()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 10)
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(42, _cts.Token);

        // Assert
        result.Should().Be(420);
    }

    [Fact]
    public async Task SingleItemAfterFilter_ShouldProcess()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x == 50)
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(100);
    }

    [Fact]
    public async Task NullableTypes_WithNull_ShouldHandle()
    {
        // Arrange - Use wrapper class to handle nullable values with notnull constraint
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<NullableInt, string>(x => x.HasValue ? x.Value!.Value.ToString() : "null")
            .Build<NullableInt, string>();

        var inputs = new[]
        {
            new NullableInt(1),
            new NullableInt(null),
            new NullableInt(3),
            new NullableInt(null),
            new NullableInt(5)
        }.ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal("1", "null", "3", "null", "5");
    }

    /// <summary>
    /// Wrapper for nullable int to satisfy notnull constraint
    /// </summary>
    private sealed record NullableInt(int? Value)
    {
        public bool HasValue => Value.HasValue;
    }

    [Fact]
    public async Task ComplexNullHandling_ShouldFilterNulls()
    {
        // Arrange - Use wrapper classes to handle nullable values with notnull constraint
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<NullableString, NullableInt>(s => new NullableInt(s.Value?.Length))
            .Filter<NullableInt>(x => x.HasValue)
            .Transform<NullableInt, int>(x => x.Value!.Value)
            .Build<NullableString, int>();

        var inputs = new[]
        {
            new NullableString("hello"),
            new NullableString(null),
            new NullableString("world"),
            new NullableString(""),
            new NullableString(null),
            new NullableString("test")
        }.ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(5, 5, 0, 4); // Lengths of non-null strings
    }

    /// <summary>
    /// Wrapper for nullable string to satisfy notnull constraint
    /// </summary>
    private sealed record NullableString(string? Value);

    #endregion

    #region Boundary Value Tests (6 tests)

    [Fact]
    public async Task MaxIntValue_ShouldHandle()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, long>(x => (long)x + 1)
            .Build<int, long>();

        // Act
        var result = await pipeline.ProcessAsync(int.MaxValue, _cts.Token);

        // Assert
        result.Should().Be((long)int.MaxValue + 1);
    }

    [Fact]
    public async Task MinIntValue_ShouldHandle()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, long>(x => (long)x - 1)
            .Build<int, long>();

        // Act
        var result = await pipeline.ProcessAsync(int.MinValue, _cts.Token);

        // Assert
        result.Should().Be((long)int.MinValue - 1);
    }

    [Fact]
    public async Task ZeroValues_ShouldHandle()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, double>(x => x == 0 ? double.NaN : 100.0 / x)
            .Build<int, double>();

        var inputs = new[] { 10, 0, 5, 0, 2 }.ToAsyncEnumerable();

        // Act
        var results = new List<double>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results[0].Should().Be(10.0);
        results[1].Should().Be(double.NaN);
        results[2].Should().Be(20.0);
        results[3].Should().Be(double.NaN);
        results[4].Should().Be(50.0);
    }

    [Fact]
    public async Task FloatSpecialValues_ShouldHandle()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<float, string>(x =>
            {
                if (float.IsNaN(x)) return "NaN";
                if (float.IsPositiveInfinity(x)) return "+Inf";
                if (float.IsNegativeInfinity(x)) return "-Inf";
                return x.ToString();
            })
            .Build<float, string>();

        var inputs = new[] { 1.0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0.0f }
            .ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal("1", "NaN", "+Inf", "-Inf", "0");
    }

    [Fact]
    public async Task VeryLargeNumbers_ShouldMaintainPrecision()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<long, decimal>(x => (decimal)x)
            .Transform<decimal, string>(x => x.ToString())
            .Build<long, string>();

        var largeNumber = 9_223_372_036_854_775_807L; // long.MaxValue

        // Act
        var result = await pipeline.ProcessAsync(largeNumber, _cts.Token);

        // Assert
        result.Should().Be("9223372036854775807");
    }

    [Fact]
    public async Task VerySmallFloats_ShouldHandle()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<float, bool>(x => x != 0)
            .Build<float, bool>();

        var inputs = new[] { 0.0f, float.Epsilon, -float.Epsilon, 1e-30f }
            .ToAsyncEnumerable();

        // Act
        var results = new List<bool>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(false, true, true, true);
    }

    #endregion

    #region Exception Handling Tests (6 tests)

    [Fact]
    public async Task DivideByZero_ShouldHandleGracefully()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                try
                {
                    return 100 / x;
                }
                catch (DivideByZeroException)
                {
                    return -1; // Error indicator
                }
            })
            .Build<int, int>();

        var inputs = new[] { 10, 0, 5, 0, 2 }.ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(10, -1, 20, -1, 50);
    }

    [Fact]
    public async Task FormatException_ShouldSkipItem()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<string, int>(s => int.Parse(s))
            .Build<string, int>();

        var inputs = new[] { "123", "abc", "456", "xyz", "789" }.ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - should skip invalid formats
        results.Should().Equal(123, 456, 789);
    }

    [Fact]
    public async Task CascadingExceptions_ShouldStopPipeline()
    {
        // Arrange
        var stage1Count = 0;
        var stage2Count = 0;

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                Interlocked.Increment(ref stage1Count);
                if (x == 5) throw new InvalidOperationException("Stage 1");
                return x + 10;
            })
            .Transform<int, int>(x =>
            {
                Interlocked.Increment(ref stage2Count);
                if (x == 20) throw new InvalidOperationException("Stage 2");
                return x * 2;
            })
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        stage1Count.Should().Be(10);
        stage2Count.Should().Be(9); // One failed in stage 1
        results.Should().HaveCount(8); // Two failures total (5 in stage1, 10 -> 20 in stage2)
    }

    [Fact]
    public async Task OutOfMemorySimulation_ShouldHandle()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, string>(x =>
            {
                if (x > 100)
                {
                    // Simulate out of memory by throwing
                    throw new OutOfMemoryException("Simulated OOM");
                }
                return x.ToString();
            })
            .Build<int, string>();

        var inputs = Enumerable.Range(1, 150).ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - should only process first 100
        results.Should().HaveCount(100);
    }

    [Fact]
    public async Task TimeoutSimulation_WithCancellation_ShouldCancel()
    {
        // Arrange
        var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                await Task.Delay(50, CancellationToken.None); // Simulate slow operation
                return x * 2;
            })
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in pipeline.ProcessManyAsync(inputs, cts.Token))
            {
                // Consume
            }
        });
    }

    [Fact]
    public async Task UnexpectedException_ShouldLogAndContinue()
    {
        // Arrange
        var unexpectedCount = 0;
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                if (x == 13)
                {
                    Interlocked.Increment(ref unexpectedCount);
                    throw new IndexOutOfRangeException("Unexpected error");
                }
                return x * 2;
            })
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 20).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        unexpectedCount.Should().Be(1);
        results.Should().HaveCount(19);
        results.Should().NotContain(26); // 13 * 2
    }

    #endregion

    #region Concurrency and Threading Tests (4 tests)

    [Fact]
    public async Task ConcurrentAccess_ShouldBeThreadSafe()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        // Act - Process from multiple tasks
        var tasks = Enumerable.Range(0, 10).Select(async i =>
        {
            var inputs = Enumerable.Range(i * 100, 100).ToAsyncEnumerable();
            var results = new List<int>();
            await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
            {
                results.Add(result);
            }
            return results;
        });

        var allResults = await Task.WhenAll(tasks);

        // Assert
        allResults.Should().AllSatisfy(results => results.Should().HaveCount(100));
        var totalCount = allResults.Sum(r => r.Count);
        totalCount.Should().Be(1000);
    }

    [Fact]
    public async Task ParallelStage_HighConcurrency_ShouldNotExceedLimit()
    {
        // Arrange
        var maxConcurrent = 0;
        var currentConcurrent = 0;
        var lockObj = new object();

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Parallel<int, int>(async x =>
            {
                var current = Interlocked.Increment(ref currentConcurrent);
                lock (lockObj)
                {
                    if (current > maxConcurrent)
                        maxConcurrent = current;
                }

                await Task.Delay(10);

                Interlocked.Decrement(ref currentConcurrent);
                return x * 2;
            }, maxConcurrency: 3)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act
        var count = 0;
        await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            count++;
        }

        // Assert
        count.Should().Be(100);
        maxConcurrent.Should().BeLessThanOrEqualTo(3);
    }

    [Fact]
    public async Task RaceCondition_WithSharedState_ShouldUseProperLocking()
    {
        // Arrange
        var counter = 0;
        var lockObj = new object();

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Tap<int>(_ =>
            {
                lock (lockObj)
                {
                    counter++;
                }
            })
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 1000).ToAsyncEnumerable();

        // Act
        await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            // Consume
        }

        // Assert
        counter.Should().Be(1000);
    }

    [Fact]
    public async Task DeadlockPrevention_NestedPipelines_ShouldNotDeadlock()
    {
        // Arrange
        var innerPipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x + 1)
            .Build<int, int>();

        var outerPipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x => await innerPipeline.ProcessAsync(x, _cts.Token))
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in outerPipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - (x + 1) * 2
        results.Should().Equal(4, 6, 8, 10, 12, 14, 16, 18, 20, 22);
    }

    #endregion

    #region Channel and Streaming Tests (4 tests)

    [Fact]
    public async Task ChannelProcessing_EmptyChannel_ShouldComplete()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        var inputChannel = Channel.CreateUnbounded<int>();
        var outputChannel = Channel.CreateUnbounded<int>();

        inputChannel.Writer.Complete(); // Empty channel

        // Act
        await pipeline.ProcessChannelAsync(
            inputChannel.Reader,
            outputChannel.Writer,
            _cts.Token);

        // Assert
        var results = new List<int>();
        await foreach (var result in outputChannel.Reader.ReadAllAsync(_cts.Token))
        {
            results.Add(result);
        }

        results.Should().BeEmpty();
    }

    [Fact]
    public async Task ChannelProcessing_LargeChannel_ShouldStreamEfficiently()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 2)
            .Filter<int>(x => x % 4 == 0)
            .Build<int, int>();

        var inputChannel = Channel.CreateUnbounded<int>();
        var outputChannel = Channel.CreateUnbounded<int>();

        // Start processing in background
        var processingTask = pipeline.ProcessChannelAsync(
            inputChannel.Reader,
            outputChannel.Writer,
            _cts.Token);

        // Act - Write data
        for (int i = 1; i <= 100; i++)
        {
            await inputChannel.Writer.WriteAsync(i, _cts.Token);
        }
        inputChannel.Writer.Complete();

        await processingTask;

        // Assert
        var results = new List<int>();
        await foreach (var result in outputChannel.Reader.ReadAllAsync(_cts.Token))
        {
            results.Add(result);
        }

        // Even numbers * 2 that are divisible by 4: 4, 8, 12, 16, ... 200
        results.Should().HaveCount(50);
        results.Should().OnlyContain(x => x % 4 == 0);
    }

    [Fact]
    public async Task ChannelProcessing_WithBackpressure_ShouldHandle()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1); // Slow processing
                return x * 2;
            })
            .Build<int, int>();

        var inputChannel = Channel.CreateBounded<int>(10); // Limited capacity
        var outputChannel = Channel.CreateBounded<int>(10);

        // Start processing
        var processingTask = pipeline.ProcessChannelAsync(
            inputChannel.Reader,
            outputChannel.Writer,
            _cts.Token);

        // Act - Write more than capacity
        var writeTask = Task.Run(async () =>
        {
            for (int i = 1; i <= 50; i++)
            {
                await inputChannel.Writer.WriteAsync(i, _cts.Token);
            }
            inputChannel.Writer.Complete();
        });

        // Read from output channel concurrently to avoid deadlock
        var readTask = Task.Run(async () =>
        {
            var count = 0;
            await foreach (var _ in outputChannel.Reader.ReadAllAsync(_cts.Token))
            {
                count++;
            }
            return count;
        });

        await writeTask;
        await processingTask;
        var resultCount = await readTask;

        // Assert
        resultCount.Should().Be(50);
    }

    [Fact]
    public async Task ChannelProcessing_WithException_ShouldCompleteChannel()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                if (x == 5) throw new InvalidOperationException("Test");
                return x * 2;
            })
            .Build<int, int>();

        var inputChannel = Channel.CreateUnbounded<int>();
        var outputChannel = Channel.CreateUnbounded<int>();

        // Start processing
        var processingTask = pipeline.ProcessChannelAsync(
            inputChannel.Reader,
            outputChannel.Writer,
            _cts.Token);

        // Act
        for (int i = 1; i <= 10; i++)
        {
            await inputChannel.Writer.WriteAsync(i, _cts.Token);
        }
        inputChannel.Writer.Complete();

        await processingTask;

        // Assert
        var results = new List<int>();
        await foreach (var result in outputChannel.Reader.ReadAllAsync(_cts.Token))
        {
            results.Add(result);
        }

        results.Should().HaveCount(9); // One failed
        results.Should().NotContain(10); // 5 * 2
    }

    #endregion

    public void Dispose()
    {
        _cts?.Cancel();
        _cts?.Dispose();
        if (_serviceProvider is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
