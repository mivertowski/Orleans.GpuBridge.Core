// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

namespace Orleans.GpuBridge.BridgeFX.Tests.Pipeline.Stages;

/// <summary>
/// Tests for <see cref="TapStage{T}"/> pipeline stage.
/// </summary>
public class TapStageTests
{
    [Fact]
    public async Task ProcessAsync_ShouldReturnSameInput()
    {
        // Arrange
        var stage = new TapStage<int>(_ => { });

        // Act
        var result = await stage.ProcessAsync(42, CancellationToken.None);

        // Assert
        result.Should().Be(42);
    }

    [Fact]
    public async Task ProcessAsync_ShouldExecuteAction()
    {
        // Arrange
        var receivedValue = 0;
        var stage = new TapStage<int>(x => receivedValue = x);

        // Act
        await stage.ProcessAsync(42, CancellationToken.None);

        // Assert
        receivedValue.Should().Be(42);
    }

    [Fact]
    public async Task ProcessAsync_ShouldExecuteAction_ForEachCall()
    {
        // Arrange
        var callCount = 0;
        var stage = new TapStage<int>(_ => callCount++);

        // Act
        await stage.ProcessAsync(1, CancellationToken.None);
        await stage.ProcessAsync(2, CancellationToken.None);
        await stage.ProcessAsync(3, CancellationToken.None);

        // Assert
        callCount.Should().Be(3);
    }

    [Fact]
    public async Task ProcessAsync_ShouldThrow_WhenInputTypeIsWrong()
    {
        // Arrange
        var stage = new TapStage<int>(_ => { });

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
        var stage = new TapStage<string>(_ => { });

        // Assert
        stage.InputType.Should().Be(typeof(string));
    }

    [Fact]
    public void OutputType_ShouldReturnSameAsInputType()
    {
        // Arrange
        var stage = new TapStage<int>(_ => { });

        // Assert
        stage.OutputType.Should().Be(typeof(int));
        stage.InputType.Should().Be(stage.OutputType);
    }

    [Fact]
    public async Task ProcessAsync_ShouldAccumulateValues()
    {
        // Arrange
        var values = new List<int>();
        var stage = new TapStage<int>(x => values.Add(x));

        // Act
        for (int i = 1; i <= 5; i++)
        {
            await stage.ProcessAsync(i, CancellationToken.None);
        }

        // Assert
        values.Should().BeEquivalentTo([1, 2, 3, 4, 5]);
    }

    [Fact]
    public async Task ProcessAsync_ShouldWork_WithComplexTypes()
    {
        // Arrange
        (string Name, int Age)? received = null;
        var stage = new TapStage<(string Name, int Age)>(x => received = x);

        // Act
        var input = ("Alice", 30);
        var result = await stage.ProcessAsync(input, CancellationToken.None);

        // Assert
        result.Should().Be(input);
        received.Should().Be(input);
    }

    [Fact]
    public async Task ProcessAsync_ShouldNotModifyInput()
    {
        // Arrange
        var original = "original value";
        string captured = "";
        var stage = new TapStage<string>(s =>
        {
            captured = s;
            // Try to modify (won't affect original since strings are immutable)
        });

        // Act
        var result = await stage.ProcessAsync(original, CancellationToken.None);

        // Assert
        result.Should().Be(original);
        captured.Should().Be(original);
    }
}
