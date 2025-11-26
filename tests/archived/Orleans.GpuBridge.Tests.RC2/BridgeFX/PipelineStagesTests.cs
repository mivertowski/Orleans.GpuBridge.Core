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
/// Comprehensive tests for individual pipeline stages.
/// Tests FilterStage, TransformStage, AsyncTransformStage, BatchStage, ParallelStage, TapStage, and KernelStage.
/// Target: 30+ tests covering all stage types and scenarios.
/// </summary>
public sealed class PipelineStagesTests : IDisposable
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IGpuBridge _bridge;
    private readonly ILogger<GpuPipeline> _logger;
    private readonly CancellationTokenSource _cts;

    public PipelineStagesTests()
    {
        var services = new ServiceCollection();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Debug));
        services.AddSingleton<IGpuBridge>(new MockGpuBridge());
        _serviceProvider = services.BuildServiceProvider();

        _bridge = _serviceProvider.GetRequiredService<IGpuBridge>();
        _logger = _serviceProvider.GetRequiredService<ILogger<GpuPipeline>>();
        _cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    }

    #region FilterStage Tests (8 tests)

    [Fact]
    public async Task FilterStage_WithMatchingPredicate_ShouldPassThrough()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x > 0)
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(5, _cts.Token);

        // Assert
        result.Should().Be(5);
    }

    [Fact]
    public async Task FilterStage_WithNonMatchingPredicate_ShouldFilterOut()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x)
            .Filter<int>(x => x > 10)
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        var inputs = new[] { 5, 15, 3, 20 }.ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - only values > 10 should pass through
        results.Should().Equal(30, 40); // 15*2, 20*2
    }

    [Fact]
    public async Task FilterStage_WithEvenPredicate_ShouldFilterEvens()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x % 2 == 0)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(2, 4, 6, 8, 10);
    }

    [Fact]
    public async Task FilterStage_MultipleFilters_ShouldApplyBoth()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x > 5)
            .Filter<int>(x => x < 15)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 20).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - should be > 5 AND < 15
        results.Should().Equal(6, 7, 8, 9, 10, 11, 12, 13, 14);
    }

    [Fact]
    public async Task FilterStage_WithComplexType_ShouldFilter()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, (int Value, bool IsEven)>(x => (x, x % 2 == 0))
            .Filter<(int Value, bool IsEven)>(x => x.IsEven)
            .Transform<(int Value, bool IsEven), int>(x => x.Value)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(2, 4, 6, 8, 10);
    }

    [Fact]
    public async Task FilterStage_WithNegativeNumbers_ShouldFilter()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x >= 0)
            .Build<int, int>();

        var inputs = new[] { -5, 3, -2, 7, 0, -1, 4 }.ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(3, 7, 0, 4);
    }

    [Fact]
    public async Task FilterStage_WithAllFiltered_ShouldReturnEmpty()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Filter<int>(x => x > 100)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 50).ToAsyncEnumerable();

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
    public async Task FilterStage_WithStringType_ShouldFilter()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, string>(x => x.ToString())
            .Filter<string>(s => s.Length > 1)
            .Build<int, string>();

        var inputs = Enumerable.Range(1, 15).ToAsyncEnumerable();

        // Act
        var results = new List<string>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert - strings "10", "11", "12", "13", "14", "15" have length > 1
        results.Should().Equal("10", "11", "12", "13", "14", "15");
    }

    #endregion

    #region TransformStage Tests (8 tests)

    [Fact]
    public async Task TransformStage_SimpleMultiply_ShouldTransform()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * 3)
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(5, _cts.Token);

        // Assert
        result.Should().Be(15);
    }

    [Fact]
    public async Task TransformStage_TypeConversion_ShouldConvert()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, float>(x => x * 1.5f)
            .Build<int, float>();

        // Act
        var result = await pipeline.ProcessAsync(10, _cts.Token);

        // Assert
        result.Should().Be(15.0f);
    }

    [Fact]
    public async Task TransformStage_ChainedTransforms_ShouldApplyAll()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x + 10)
            .Transform<int, int>(x => x * 2)
            .Transform<int, int>(x => x - 5)
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(5, _cts.Token);

        // Assert - (5 + 10) * 2 - 5 = 25
        result.Should().Be(25);
    }

    [Fact]
    public async Task TransformStage_ToComplexType_ShouldCreate()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, (int Original, int Doubled, int Squared)>(x => (x, x * 2, x * x))
            .Build<int, (int Original, int Doubled, int Squared)>();

        // Act
        var result = await pipeline.ProcessAsync(5, _cts.Token);

        // Assert
        result.Should().Be((5, 10, 25));
    }

    [Fact]
    public async Task TransformStage_FromComplexType_ShouldExtract()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<(int X, int Y), int>(tuple => tuple.X + tuple.Y)
            .Build<(int X, int Y), int>();

        // Act
        var result = await pipeline.ProcessAsync((10, 20), _cts.Token);

        // Assert
        result.Should().Be(30);
    }

    [Fact]
    public async Task TransformStage_WithBatchProcessing_ShouldTransformAll()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x * x)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(1, 4, 9, 16, 25, 36, 49, 64, 81, 100);
    }

    [Fact]
    public async Task TransformStage_WithNegativeValues_ShouldTransform()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => Math.Abs(x))
            .Build<int, int>();

        var inputs = new[] { -5, 3, -2, 7, -10 }.ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        results.Should().Equal(5, 3, 2, 7, 10);
    }

    [Fact]
    public async Task TransformStage_StringManipulation_ShouldTransform()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, string>(x => $"Value: {x}")
            .Transform<string, string>(s => s.ToUpper())
            .Build<int, string>();

        // Act
        var result = await pipeline.ProcessAsync(42, _cts.Token);

        // Assert
        result.Should().Be("VALUE: 42");
    }

    #endregion

    #region AsyncTransformStage Tests (8 tests)

    [Fact]
    public async Task AsyncTransformStage_SimpleAsync_ShouldTransform()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1, _cts.Token);
                return x * 2;
            })
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(10, _cts.Token);

        // Assert
        result.Should().Be(20);
    }

    [Fact]
    public async Task AsyncTransformStage_WithCancellation_ShouldRespectToken()
    {
        // Arrange
        var cts = new CancellationTokenSource();
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                cts.Cancel(); // Cancel during processing
                await Task.Delay(100, CancellationToken.None);
                return x * 2;
            })
            .Build<int, int>();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await pipeline.ProcessAsync(10, cts.Token));
    }

    [Fact]
    public async Task AsyncTransformStage_ChainedAsync_ShouldExecuteSequentially()
    {
        // Arrange
        var executionOrder = new List<int>();
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1);
                executionOrder.Add(1);
                return x + 10;
            })
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1);
                executionOrder.Add(2);
                return x * 2;
            })
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1);
                executionOrder.Add(3);
                return x - 5;
            })
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(5, _cts.Token);

        // Assert
        result.Should().Be(25); // (5 + 10) * 2 - 5
        executionOrder.Should().Equal(1, 2, 3);
    }

    [Fact]
    public async Task AsyncTransformStage_WithIOOperation_ShouldComplete()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, string>(async x =>
            {
                await Task.Delay(1);
                return $"Processed: {x}";
            })
            .Build<int, string>();

        // Act
        var result = await pipeline.ProcessAsync(42, _cts.Token);

        // Assert
        result.Should().Be("Processed: 42");
    }

    [Fact]
    public async Task AsyncTransformStage_BatchProcessing_ShouldProcessAll()
    {
        // Arrange
        var processedCount = 0;
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1);
                Interlocked.Increment(ref processedCount);
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
        results.Should().HaveCount(20);
        processedCount.Should().Be(20);
        results.Should().BeInAscendingOrder();
    }

    [Fact]
    public async Task AsyncTransformStage_WithException_ShouldPropagateError()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(async x =>
            {
                await Task.Delay(1);
                if (x == 5) throw new InvalidOperationException("Test error");
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

        // Assert - should skip the failed item
        results.Should().HaveCount(9);
        results.Should().NotContain(10); // 5 * 2 would be 10
    }

    [Fact]
    public async Task AsyncTransformStage_MixedWithSync_ShouldWork()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => x + 1) // Sync
            .Transform<int, int>(async x => // Async
            {
                await Task.Delay(1);
                return x * 2;
            })
            .Transform<int, int>(x => x - 3) // Sync
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(5, _cts.Token);

        // Assert - (5 + 1) * 2 - 3 = 9
        result.Should().Be(9);
    }

    [Fact]
    public async Task AsyncTransformStage_WithTaskResult_ShouldUnwrap()
    {
        // Arrange
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, int>(x => Task.FromResult(x * 3))
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(7, _cts.Token);

        // Assert
        result.Should().Be(21);
    }

    #endregion

    #region TapStage Tests (6 tests)

    [Fact]
    public async Task TapStage_ShouldExecuteSideEffect()
    {
        // Arrange
        var sideEffectValue = 0;
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Tap<int>(x => sideEffectValue = x * 2)
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(10, _cts.Token);

        // Assert
        result.Should().Be(10); // Tap doesn't modify value
        sideEffectValue.Should().Be(20); // Side effect executed
    }

    [Fact]
    public async Task TapStage_MultipleTaps_ShouldExecuteAll()
    {
        // Arrange
        var tap1Value = 0;
        var tap2Value = 0;
        var tap3Value = 0;

        var pipeline = new GpuPipeline(_bridge, _logger)
            .Tap<int>(x => tap1Value = x)
            .Transform<int, int>(x => x * 2)
            .Tap<int>(x => tap2Value = x)
            .Transform<int, int>(x => x + 5)
            .Tap<int>(x => tap3Value = x)
            .Build<int, int>();

        // Act
        var result = await pipeline.ProcessAsync(10, _cts.Token);

        // Assert
        result.Should().Be(25); // (10 * 2) + 5
        tap1Value.Should().Be(10);
        tap2Value.Should().Be(20);
        tap3Value.Should().Be(25);
    }

    [Fact]
    public async Task TapStage_WithLogging_ShouldLog()
    {
        // Arrange
        var loggedValues = new List<int>();
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Tap<int>(x => loggedValues.Add(x))
            .Transform<int, int>(x => x * 2)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 5).ToAsyncEnumerable();

        // Act
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // Assert
        loggedValues.Should().Equal(1, 2, 3, 4, 5);
        results.Should().Equal(2, 4, 6, 8, 10);
    }

    [Fact]
    public async Task TapStage_WithCounter_ShouldCount()
    {
        // Arrange
        var counter = 0;
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Tap<int>(_ => Interlocked.Increment(ref counter))
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 100).ToAsyncEnumerable();

        // Act
        await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            // Just consume
        }

        // Assert
        counter.Should().Be(100);
    }

    [Fact]
    public async Task TapStage_WithComplexType_ShouldAccess()
    {
        // Arrange
        var capturedNames = new List<string>();
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Transform<int, (int Id, string Name)>(x => (x, $"Item{x}"))
            .Tap<(int Id, string Name)>(t => capturedNames.Add(t.Name))
            .Transform<(int Id, string Name), int>(t => t.Id)
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 5).ToAsyncEnumerable();

        // Act
        await foreach (var _ in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            // Just consume
        }

        // Assert
        capturedNames.Should().Equal("Item1", "Item2", "Item3", "Item4", "Item5");
    }

    [Fact]
    public async Task TapStage_ThatThrows_ShouldNotBreakPipeline()
    {
        // Arrange
        var successCount = 0;
        var pipeline = new GpuPipeline(_bridge, _logger)
            .Tap<int>(x =>
            {
                if (x == 5) throw new InvalidOperationException("Tap error");
                Interlocked.Increment(ref successCount);
            })
            .Build<int, int>();

        var inputs = Enumerable.Range(1, 10).ToAsyncEnumerable();

        // Act & Assert - Tap exceptions should propagate and cause processing to fail for that item
        var results = new List<int>();
        await foreach (var result in pipeline.ProcessManyAsync(inputs, _cts.Token))
        {
            results.Add(result);
        }

        // The item where tap threw should be skipped
        results.Should().HaveCount(9);
        successCount.Should().Be(9);
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
