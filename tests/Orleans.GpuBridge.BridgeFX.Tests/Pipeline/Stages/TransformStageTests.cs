// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

namespace Orleans.GpuBridge.BridgeFX.Tests.Pipeline.Stages;

/// <summary>
/// Tests for <see cref="TransformStage{TIn, TOut}"/> pipeline stage.
/// </summary>
public class TransformStageTests
{
    [Fact]
    public async Task ProcessAsync_ShouldTransformInput()
    {
        // Arrange
        var stage = new TransformStage<int, string>(x => x.ToString());

        // Act
        var result = await stage.ProcessAsync(42, CancellationToken.None);

        // Assert
        result.Should().Be("42");
    }

    [Fact]
    public async Task ProcessAsync_ShouldTransformToAnotherType()
    {
        // Arrange
        var stage = new TransformStage<string, int>(s => s.Length);

        // Act
        var result = await stage.ProcessAsync("hello", CancellationToken.None);

        // Assert
        result.Should().Be(5);
    }

    [Fact]
    public async Task ProcessAsync_ShouldSupportComplexTransformation()
    {
        // Arrange
        var stage = new TransformStage<int, (int, string)>(x => (x, x.ToString()));

        // Act
        var result = await stage.ProcessAsync(42, CancellationToken.None);

        // Assert
        result.Should().Be((42, "42"));
    }

    [Fact]
    public async Task ProcessAsync_ShouldThrow_WhenInputTypeIsWrong()
    {
        // Arrange
        var stage = new TransformStage<int, string>(x => x.ToString());

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
        var stage = new TransformStage<int, string>(x => x.ToString());

        // Assert
        stage.InputType.Should().Be(typeof(int));
    }

    [Fact]
    public void OutputType_ShouldReturnCorrectType()
    {
        // Arrange
        var stage = new TransformStage<int, string>(x => x.ToString());

        // Assert
        stage.OutputType.Should().Be(typeof(string));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(-100)]
    [InlineData(int.MaxValue)]
    public async Task ProcessAsync_ShouldTransformVariousInputs(int input)
    {
        // Arrange
        var stage = new TransformStage<int, int>(x => x * 2);

        // Act
        var result = await stage.ProcessAsync(input, CancellationToken.None);

        // Assert
        result.Should().Be(input * 2);
    }

    [Fact]
    public async Task ProcessAsync_ShouldHandleNullTransformResult()
    {
        // Arrange
        var stage = new TransformStage<string, string?>(s => s.Length > 3 ? s : null);

        // Act
        var shortResult = await stage.ProcessAsync("hi", CancellationToken.None);
        var longResult = await stage.ProcessAsync("hello", CancellationToken.None);

        // Assert
        shortResult.Should().BeNull();
        longResult.Should().Be("hello");
    }

    [Fact]
    public async Task ProcessAsync_ShouldChainTransformations()
    {
        // Arrange
        var stage1 = new TransformStage<int, string>(x => x.ToString());
        var stage2 = new TransformStage<string, int>(s => s.Length);

        // Act
        var intermediate = await stage1.ProcessAsync(12345, CancellationToken.None);
        var final = await stage2.ProcessAsync(intermediate!, CancellationToken.None);

        // Assert
        intermediate.Should().Be("12345");
        final.Should().Be(5);
    }
}
