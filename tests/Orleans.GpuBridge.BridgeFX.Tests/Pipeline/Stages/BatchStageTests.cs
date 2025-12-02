// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

namespace Orleans.GpuBridge.BridgeFX.Tests.Pipeline.Stages;

/// <summary>
/// Tests for <see cref="BatchStage{T}"/> pipeline stage.
/// </summary>
public class BatchStageTests
{
    [Fact]
    public async Task ProcessAsync_ShouldReturnNull_BeforeBatchSizeReached()
    {
        // Arrange
        var stage = new BatchStage<int>(5, TimeSpan.FromHours(1));

        // Act
        var result = await stage.ProcessAsync(1, CancellationToken.None);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task ProcessAsync_ShouldReturnBatch_WhenBatchSizeReached()
    {
        // Arrange
        var stage = new BatchStage<int>(3, TimeSpan.FromHours(1));

        // Act
        await stage.ProcessAsync(1, CancellationToken.None);
        await stage.ProcessAsync(2, CancellationToken.None);
        var result = await stage.ProcessAsync(3, CancellationToken.None);

        // Assert
        result.Should().NotBeNull();
        result.Should().BeAssignableTo<IReadOnlyList<int>>();
        var batch = (IReadOnlyList<int>)result!;
        batch.Should().HaveCount(3);
        batch.Should().ContainInOrder(1, 2, 3);
    }

    [Fact]
    public async Task ProcessAsync_ShouldResetBuffer_AfterFlush()
    {
        // Arrange
        var stage = new BatchStage<int>(2, TimeSpan.FromHours(1));

        // Act - first batch
        await stage.ProcessAsync(1, CancellationToken.None);
        var firstBatch = await stage.ProcessAsync(2, CancellationToken.None);

        // Act - second batch starts fresh
        var secondBatchPartial = await stage.ProcessAsync(3, CancellationToken.None);

        // Assert
        firstBatch.Should().NotBeNull();
        secondBatchPartial.Should().BeNull();
    }

    [Fact]
    public async Task ProcessAsync_ShouldThrow_WhenInputTypeIsWrong()
    {
        // Arrange
        var stage = new BatchStage<int>(5, TimeSpan.FromHours(1));

        // Act
        var act = () => stage.ProcessAsync("not an int", CancellationToken.None);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*Expected*Int32*got*String*");
    }

    [Fact]
    public void InputType_ShouldReturnCorrectType()
    {
        // Arrange
        var stage = new BatchStage<string>(5, TimeSpan.FromHours(1));

        // Assert
        stage.InputType.Should().Be(typeof(string));
    }

    [Fact]
    public void OutputType_ShouldReturnReadOnlyList()
    {
        // Arrange
        var stage = new BatchStage<int>(5, TimeSpan.FromHours(1));

        // Assert
        stage.OutputType.Should().Be(typeof(IReadOnlyList<int>));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public async Task ProcessAsync_ShouldRespectBatchSize(int batchSize)
    {
        // Arrange
        var stage = new BatchStage<int>(batchSize, TimeSpan.FromHours(1));

        // Act
        object? result = null;
        for (int i = 1; i <= batchSize; i++)
        {
            result = await stage.ProcessAsync(i, CancellationToken.None);
        }

        // Assert
        result.Should().NotBeNull();
        var batch = (IReadOnlyList<int>)result!;
        batch.Should().HaveCount(batchSize);
    }

    [Fact]
    public async Task ProcessAsync_ShouldFlushOnTimeout()
    {
        // Arrange - tiny timeout
        var stage = new BatchStage<int>(100, TimeSpan.Zero);

        // Act - add one item
        // The timeout check happens on each ProcessAsync
        // With TimeSpan.Zero, each call should trigger a flush
        var result = await stage.ProcessAsync(1, CancellationToken.None);

        // Assert - should flush immediately due to timeout
        result.Should().NotBeNull();
        var batch = (IReadOnlyList<int>)result!;
        batch.Should().HaveCount(1);
    }

    [Fact]
    public async Task ProcessAsync_ShouldCollectMultipleBatches()
    {
        // Arrange
        var stage = new BatchStage<int>(3, TimeSpan.FromHours(1));
        var batches = new List<IReadOnlyList<int>>();

        // Act - process 9 items (3 batches of 3)
        for (int i = 1; i <= 9; i++)
        {
            var result = await stage.ProcessAsync(i, CancellationToken.None);
            if (result is IReadOnlyList<int> batch)
            {
                batches.Add(batch);
            }
        }

        // Assert
        batches.Should().HaveCount(3);
        batches[0].Should().ContainInOrder(1, 2, 3);
        batches[1].Should().ContainInOrder(4, 5, 6);
        batches[2].Should().ContainInOrder(7, 8, 9);
    }
}
