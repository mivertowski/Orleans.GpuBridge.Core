// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

namespace Orleans.GpuBridge.BridgeFX.Tests.Pipeline.Stages;

/// <summary>
/// Tests for <see cref="FilterStage{T}"/> pipeline stage.
/// </summary>
public class FilterStageTests
{
    [Fact]
    public async Task ProcessAsync_ShouldReturnInput_WhenPredicatePasses()
    {
        // Arrange
        var stage = new FilterStage<int>(x => x > 0);

        // Act
        var result = await stage.ProcessAsync(42, CancellationToken.None);

        // Assert
        result.Should().Be(42);
    }

    [Fact]
    public async Task ProcessAsync_ShouldReturnNull_WhenPredicateFails()
    {
        // Arrange
        var stage = new FilterStage<int>(x => x > 100);

        // Act
        var result = await stage.ProcessAsync(42, CancellationToken.None);

        // Assert
        result.Should().BeNull();
    }

    [Fact]
    public async Task ProcessAsync_ShouldFilter_WithStringPredicate()
    {
        // Arrange
        var stage = new FilterStage<string>(s => s.Length > 3);

        // Act
        var passResult = await stage.ProcessAsync("hello", CancellationToken.None);
        var failResult = await stage.ProcessAsync("hi", CancellationToken.None);

        // Assert
        passResult.Should().Be("hello");
        failResult.Should().BeNull();
    }

    [Fact]
    public async Task ProcessAsync_ShouldThrow_WhenInputTypeIsWrong()
    {
        // Arrange
        var stage = new FilterStage<int>(x => x > 0);

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
        var stage = new FilterStage<int>(x => true);

        // Assert
        stage.InputType.Should().Be(typeof(int));
    }

    [Fact]
    public void OutputType_ShouldReturnSameAsInputType()
    {
        // Arrange
        var stage = new FilterStage<string>(s => true);

        // Assert
        stage.OutputType.Should().Be(typeof(string));
        stage.InputType.Should().Be(stage.OutputType);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, true)]
    [InlineData(-1, false)]
    [InlineData(100, true)]
    public async Task ProcessAsync_ShouldApplyPredicateCorrectly(int input, bool shouldPass)
    {
        // Arrange
        var stage = new FilterStage<int>(x => x > 0);

        // Act
        var result = await stage.ProcessAsync(input, CancellationToken.None);

        // Assert
        if (shouldPass)
        {
            result.Should().Be(input);
        }
        else
        {
            result.Should().BeNull();
        }
    }

    [Fact]
    public async Task ProcessAsync_ShouldFilter_WithComplexPredicate()
    {
        // Arrange
        var stage = new FilterStage<int>(x => x % 2 == 0 && x > 10);

        // Act
        var results = new List<object?>();
        for (int i = 0; i < 20; i++)
        {
            results.Add(await stage.ProcessAsync(i, CancellationToken.None));
        }

        // Assert
        var nonNullResults = results.Where(r => r != null).Cast<int>().ToList();
        nonNullResults.Should().BeEquivalentTo([12, 14, 16, 18]);
    }
}
