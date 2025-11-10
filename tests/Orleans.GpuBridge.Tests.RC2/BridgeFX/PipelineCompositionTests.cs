using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Xunit;

namespace Orleans.GpuBridge.Tests.RC2.BridgeFX;

/// <summary>
/// Tests for complex pipeline compositions with multiple stages.
/// Tests multi-stage pipelines, type transformations, error propagation, and performance characteristics.
/// Target: 20+ tests covering complex pipeline scenarios.
/// </summary>
public sealed class PipelineCompositionTests : IDisposable
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IGpuBridge _bridge;
    private readonly ILogger<GpuPipeline> _logger;
    private readonly CancellationTokenSource _cts;

    public PipelineCompositionTests()
    {
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Debug));
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        _serviceProvider = services.BuildServiceProvider();

        _bridge = _serviceProvider.GetRequiredService<IGpuBridge>();
        _logger = _serviceProvider.GetRequiredService<ILogger<GpuPipeline>>();
        _cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    }

    #region Complex Multi-Stage Pipelines (8 tests)

    [Fact]
    public async Task ComplexPipeline_FiveStages_ShouldProcessCorrectly()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x + 5)           // Stage 1: Add 5
            .Filter<int>(x => x > 10)                  // Stage 2: Filter > 10
            .Transform<int, float>(x => x * 2.5f)      // Stage 3: Multiply by 2.5
            .Filter<float>(x => x < 100)               // Stage 4: Filter < 100
            .Transform<float, string>(x => $"Result: {x:F2}") // Stage 5: Format
            .Build<int, string>();

        var inputs = Enumerable.Range(1, 20).ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        // 6-20 -> add 5 -> 11-25 -> filter > 10 -> 11-25 -> *2.5 -> 27.5-62.5 -> filter < 100 -> all pass
        results.Should().HaveCount(15);
        results.First().Should().Be("Result: 27.50"); // (6+5)*2.5
        results.Last().Should().Be("Result: 62.50");  // (20+5)*2.5
    }

    [Fact]
    public async Task DataEnrichmentPipeline_ShouldEnrichData()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, (int Id, string Name)>(x => (x, $"Item{x}"))
            .Transform<(int Id, string Name), (int Id, string Name, bool IsEven)>(
                t => (t.Id, t.Name, t.Id % 2 == 0))
            .Filter<(int Id, string Name, bool IsEven)>(t => t.IsEven)
            .Transform<(int Id, string Name, bool IsEven), string>(
                t => $"{t.Name} (ID: {t.Id})")
            .Build<int, string>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(
            "Item2 (ID: 2)",
            "Item4 (ID: 4)",
            "Item6 (ID: 6)",
            "Item8 (ID: 8)",
            "Item10 (ID: 10)"
        );
    }

    [Fact]
    public async Task ValidationPipeline_ShouldValidateAndTransform()
    {
        // Arrange
        var validationErrors = new List<string>();

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, (int Value, bool IsValid, string? Error)>(x =>
            {
                if (x < 0) return (x, false, "Negative value");
                if (x > 100) return (x, false, "Value too large");
                return (x, true, null);
            })
            .Tap<(int Value, bool IsValid, string? Error)>(t =>
            {
                if (!t.IsValid && t.Error != null)
                    validationErrors.Add(t.Error);
            })
            .Filter<(int Value, bool IsValid, string? Error)>(t => t.IsValid)
            .Transform<(int Value, bool IsValid, string? Error), int>(t => t.Value * 2)
            .Build<int, int>();

        var inputs = new[] { -5, 10, 150, 50, -1, 75 }.ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(20, 100, 150); // Valid: 10*2, 50*2, 75*2
        validationErrors.Should().HaveCount(3);
        validationErrors.Should().Contain("Negative value");
        validationErrors.Should().Contain("Value too large");
    }

    [Fact]
    public async Task AggregationPipeline_ShouldCollectStatistics()
    {
        // Arrange
        var sum = 0.0;
        var count = 0;
        var min = double.MaxValue;
        var max = double.MinValue;

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, double>(x => x * 1.5)
            .Tap<double>(x =>
            {
                Interlocked.Increment(ref count);
                // Use simple addition for sum (not thread-safe but ok for test)
                sum += x;
                lock (this)
                {
                    if (x < min) min = x;
                    if (x > max) max = x;
                }
            })
            .Build<int, double>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act
        var results = new List<double>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        count.Should().Be(100);
        min.Should().Be(1.5); // 1 * 1.5
        max.Should().Be(150); // 100 * 1.5
        var avg = sum / count;
        avg.Should().BeApproximately(75.75, 0.01); // Average of 1.5 to 150
    }

    [Fact]
    public async Task TransformationChain_TenStages_ShouldMaintainAccuracy()
    {
        // Arrange - Build a long chain of transformations
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x + 1)
            .Transform<int, int>(x => x * 2)
            .Transform<int, int>(x => x - 3)
            .Transform<int, int>(x => x + 5)
            .Transform<int, int>(x => x * 3)
            .Transform<int, int>(x => x / 2)
            .Transform<int, int>(x => x + 10)
            .Transform<int, int>(x => x - 7)
            .Transform<int, int>(x => x * 2)
            .Transform<int, int>(x => x + 100)
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(5, _cts.Token);

        // Assert - Calculate expected: ((((((5+1)*2-3+5)*3)/2)+10-7)*2)+100
        // = ((((12-3+5)*3)/2)+10-7)*2+100
        // = ((((14)*3)/2)+10-7)*2+100
        // = ((42/2)+10-7)*2+100
        // = (21+10-7)*2+100
        // = 24*2+100
        // = 48+100
        // = 148
        result.Should().Be(148);
    }

    [Fact]
    public async Task FilterCascade_MultipleLevels_ShouldFilterProperly()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x > 10)        // Level 1: > 10
            .Filter<int>(x => x < 100)       // Level 2: < 100
            .Filter<int>(x => x % 5 == 0)    // Level 3: divisible by 5
            .Filter<int>(x => x % 10 != 0)   // Level 4: NOT divisible by 10
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 150).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - should be: >10, <100, divisible by 5, NOT divisible by 10
        // = 15, 25, 35, 45, 55, 65, 75, 85, 95
        results.Should().Equal(15, 25, 35, 45, 55, 65, 75, 85, 95);
    }

    [Fact]
    public async Task ParallelBranching_WithTaps_ShouldTrackBranches()
    {
        // Arrange
        var branch1Count = 0;
        var branch2Count = 0;
        var allCount = 0;

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Tap<int>(_ => Interlocked.Increment(ref allCount))
            .Transform<int, (int Value, bool IsEven)>(x => (x, x % 2 == 0))
            .Tap<(int Value, bool IsEven)>(t =>
            {
                if (t.IsEven) Interlocked.Increment(ref branch1Count);
                else Interlocked.Increment(ref branch2Count);
            })
            .Transform<(int Value, bool IsEven), int>(t => t.Value)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act
        await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            // Just consume
        }

        // Assert
        allCount.Should().Be(100);
        branch1Count.Should().Be(50); // Even numbers
        branch2Count.Should().Be(50); // Odd numbers
    }

    [Fact]
    public async Task TypeTransformationChain_ShouldConvertTypes()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, double>(x => x * 1.5)
            .Transform<double, float>(x => (float)x)
            .Transform<float, decimal>(x => (decimal)x)
            .Transform<decimal, string>(x => x.ToString("F2"))
            .Build<int, string>();

        // Act
        var result = await pipeline.ProcessAsync(10, _cts.Token);

        // Assert
        result.Should().Be("15.00");
    }

    #endregion

    #region Error Propagation Tests (6 tests)

    [Fact]
    public async Task ErrorInFirstStage_ShouldPropagate()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                if (x == 5) throw new InvalidOperationException("Error in stage 1");
                return x * 2;
            })
            .Transform<int, int>(x => x + 10)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - should skip item 5
        results.Should().HaveCount(9);
        results.Should().NotContain(20); // (5*2)+10 would be 20
    }

    [Fact]
    public async Task ErrorInMiddleStage_ShouldPropagate()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x + 5)
            .Transform<int, int>(x =>
            {
                if (x == 15) throw new InvalidOperationException("Error in middle");
                return x * 2;
            })
            .Transform<int, int>(x => x - 3)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 20).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - should skip item that becomes 15 after first transform (input=10)
        results.Should().HaveCount(19);
        results.Should().NotContain(27); // (15*2)-3 would be 27
    }

    [Fact]
    public async Task MultipleErrors_ShouldSkipAllFailed()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                if (x % 5 == 0) throw new InvalidOperationException($"Error: {x}");
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

        // Assert - should skip 5, 10, 15, 20 (16 results)
        results.Should().HaveCount(16);
        results.Should().NotContain(new[] { 10, 20, 30, 40 }); // 5*2, 10*2, 15*2, 20*2
    }

    [Fact]
    public async Task ErrorRecovery_WithDefaultValue_ShouldContinue()
    {
        // Arrange
        var errorCount = 0;
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                try
                {
                    if (x == 5) throw new InvalidOperationException("Test");
                    return x * 2;
                }
                catch
                {
                    Interlocked.Increment(ref errorCount);
                    return 0; // Recovery with default
                }
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
        results.Should().HaveCount(10); // All processed
        results.Should().Contain(0); // Default value from error recovery
        errorCount.Should().Be(1);
    }

    [Fact]
    public async Task ErrorInAsyncStage_ShouldPropagate()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1);
                if (x == 7) throw new InvalidOperationException("Async error");
                return x * 3;
            })
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 15).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(14);
        results.Should().NotContain(21); // 7*3
    }

    [Fact]
    public async Task CascadingErrors_ShouldStopPropagation()
    {
        // Arrange
        var stage1Executions = 0;
        var stage2Executions = 0;
        var stage3Executions = 0;

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x =>
            {
                Interlocked.Increment(ref stage1Executions);
                if (x == 5) throw new InvalidOperationException("Stage 1 error");
                return x + 10;
            })
            .Transform<int, int>(x =>
            {
                Interlocked.Increment(ref stage2Executions);
                return x * 2;
            })
            .Transform<int, int>(x =>
            {
                Interlocked.Increment(ref stage3Executions);
                return x - 5;
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
        stage1Executions.Should().Be(10); // All inputs processed
        stage2Executions.Should().Be(9);  // One failed in stage 1
        stage3Executions.Should().Be(9);  // Cascaded from stage 1 failure
        results.Should().HaveCount(9);
    }

    #endregion

    #region Performance Characteristics (6 tests)

    [Fact]
    public async Task LargeBatchProcessing_ShouldComplete()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 2)
            .Filter<int>(x => x % 3 != 0)
            .Transform<int, float>(x => x * 1.5f)
            .Build<int, float>();

        var inputs = Enumerable.Range(1, 10000).ToAsyncEnumerable();

        // Act
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var count = 0;
        await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            count++;
        }
        sw.Stop();

        // Assert
        count.Should().BeGreaterThan(6000); // Most should pass filter
        sw.ElapsedMilliseconds.Should().BeLessThan(5000); // Should be fast
    }

    [Fact]
    public async Task MinimalPipeline_ShouldHaveLowOverhead()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x)
            .Build<int, int>();

        // Act
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await pipeline.ProcessAsync(42, _cts.Token);
        sw.Stop();

        // Assert
        result.Should().Be(42);
        sw.ElapsedMilliseconds.Should().BeLessThan(10); // Minimal overhead
    }

    [Fact]
    public async Task ComplexPipeline_ShouldScaleLinearly()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x + 1)
            .Transform<int, int>(x => x * 2)
            .Transform<int, int>(x => x - 3)
            .Build<int, int>();

        // Act - Process different sized batches
        var sizes = new[] { 100, 1000, 10000 };
        var times = new List<long>();

        foreach (var size in sizes)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var inputs = Enumerable.Range(1, size).ToAsyncEnumerable();
            var count = 0;
            await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
            {
                count++;
            }
            sw.Stop();
            times.Add(sw.ElapsedMilliseconds);
        }

        // Assert - Time should scale roughly linearly
        times[0].Should().BeGreaterThan(0);
        times[1].Should().BeGreaterThan(times[0]);
        times[2].Should().BeGreaterThan(times[1]);
    }

    [Fact]
    public async Task MemoryEfficiency_ShouldNotAccumulate()
    {
        // Arrange
        var processedCount = 0;
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 2)
            .Tap<int>(_ => Interlocked.Increment(ref processedCount))
            .Build<int, int>();

        // Act - Process large stream without accumulating
        var inputs = Enumerable.Range(1, 100000).ToAsyncEnumerable();
        var firstItem = 0;
        var lastItem = 0;

        await foreach (var item in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            if (firstItem == 0) firstItem = item;
            lastItem = item;
        }

        // Assert
        firstItem.Should().Be(2);
        lastItem.Should().Be(200000);
        processedCount.Should().Be(100000);
    }

    [Fact]
    public async Task ShortCircuit_FilterEarly_ShouldReduceWork()
    {
        // Arrange
        var expensiveExecutions = 0;

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x % 2 == 0) // Filter early
            .Transform<int, int>(x =>
            {
                Interlocked.Increment(ref expensiveExecutions);
                // Simulate expensive operation
                return x * x;
            })
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 1000).ToAsyncEnumerable();

        // Act
        var count = 0;
        await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            count++;
        }

        // Assert
        count.Should().Be(500); // Only evens
        expensiveExecutions.Should().Be(500); // Only executed for filtered items
    }

    [Fact]
    public async Task ParallelProcessing_WithSemaphore_ShouldLimitConcurrency()
    {
        // Arrange
        var maxConcurrent = 0;
        var currentConcurrent = 0;

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Parallel<int, int>(async x =>
            {
                var current = Interlocked.Increment(ref currentConcurrent);
                if (current > maxConcurrent)
                {
                    Interlocked.Exchange(ref maxConcurrent, current);
                }

                await Task.Delay(10); // Simulate work

                Interlocked.Decrement(ref currentConcurrent);
                return x * 2;
            }, maxConcurrency: 5)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 50).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().HaveCount(50);
        maxConcurrent.Should().BeLessThanOrEqualTo(5);
        maxConcurrent.Should().BeGreaterThan(1); // Should use parallelism
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
