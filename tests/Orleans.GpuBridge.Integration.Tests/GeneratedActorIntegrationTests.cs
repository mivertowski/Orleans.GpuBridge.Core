// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Grains.Base;
using Orleans.GpuBridge.Grains.Generated;
using Orleans.Runtime;
using Xunit;

namespace Orleans.GpuBridge.Integration.Tests;

/// <summary>
/// End-to-end integration tests for generated GPU-native actors.
/// Validates the complete flow from interface method call through generated grain
/// to CPU fallback execution (GPU execution tested separately with DotCompute).
/// </summary>
public class GeneratedActorIntegrationTests
{
    /// <summary>
    /// Tests that AddAsync returns correct sum via generated grain.
    /// </summary>
    [Fact]
    public async Task AddAsync_ReturnsSumOfTwoIntegers()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        var result = await grain.AddAsync(3, 5);

        // Assert
        Assert.Equal(8, result);
    }

    /// <summary>
    /// Tests that SubtractAsync returns correct difference.
    /// </summary>
    [Fact]
    public async Task SubtractAsync_ReturnsDifferenceOfTwoIntegers()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        var result = await grain.SubtractAsync(10, 4);

        // Assert
        Assert.Equal(6, result);
    }

    /// <summary>
    /// Tests that MultiplyAsync accumulates product in state.
    /// </summary>
    [Fact]
    public async Task MultiplyAsync_AccumulatesProductInState()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        await grain.MultiplyAsync(3, 4); // 12
        await grain.MultiplyAsync(2, 5); // 10
        var accumulator = await grain.GetAccumulatorAsync();

        // Assert
        Assert.Equal(22, accumulator); // 12 + 10
    }

    /// <summary>
    /// Tests that FactorialAsync computes correct factorial.
    /// </summary>
    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(5, 120)]
    [InlineData(10, 3628800)]
    [InlineData(12, 479001600)]
    public async Task FactorialAsync_ComputesCorrectFactorial(int n, long expected)
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        var result = await grain.FactorialAsync(n);

        // Assert
        Assert.Equal(expected, result);
    }

    /// <summary>
    /// Tests that GetAccumulatorAsync returns initial value of zero.
    /// </summary>
    [Fact]
    public async Task GetAccumulatorAsync_ReturnsZeroInitially()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        var result = await grain.GetAccumulatorAsync();

        // Assert
        Assert.Equal(0, result);
    }

    /// <summary>
    /// Tests that ResetAsync resets accumulator to zero.
    /// </summary>
    [Fact]
    public async Task ResetAsync_ResetsAccumulatorToZero()
    {
        // Arrange
        var grain = CreateCalculatorGrain();
        await grain.MultiplyAsync(5, 10); // Accumulator = 50

        // Act
        await grain.ResetAsync();
        var accumulator = await grain.GetAccumulatorAsync();

        // Assert
        Assert.Equal(0, accumulator);
    }

    /// <summary>
    /// Tests multiple operations in sequence.
    /// </summary>
    [Fact]
    public async Task MultipleOperations_MaintainCorrectState()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        var sum = await grain.AddAsync(10, 20);
        var diff = await grain.SubtractAsync(sum, 5);
        await grain.MultiplyAsync(diff, 2);
        var factorial = await grain.FactorialAsync(5);
        var accumulator = await grain.GetAccumulatorAsync();

        // Assert
        Assert.Equal(30, sum);
        Assert.Equal(25, diff);
        Assert.Equal(50, accumulator); // 25 * 2
        Assert.Equal(120, factorial);
    }

    /// <summary>
    /// Tests that telemetry is correctly updated.
    /// </summary>
    [Fact]
    public async Task GetTelemetry_ReturnsCorrectMetrics()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act - perform several operations
        await grain.AddAsync(1, 2);
        await grain.SubtractAsync(5, 3);
        await grain.MultiplyAsync(2, 2);
        await grain.FactorialAsync(5);

        var telemetry = grain.GetTelemetry();

        // Assert
        Assert.Equal("CalculatorActorGrain", telemetry.ActorType);
        Assert.Equal(4, telemetry.TotalMessages);
        Assert.Equal(4, telemetry.CpuFallbacks); // No GPU in tests
        Assert.Equal(0, telemetry.GpuExecutions);
        Assert.False(telemetry.IsGpuAvailable);
    }

    /// <summary>
    /// Tests handler metrics tracking.
    /// </summary>
    [Fact]
    public async Task GetHandlerMetrics_TracksInvocationCounts()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        await grain.AddAsync(1, 1);
        await grain.AddAsync(2, 2);
        await grain.AddAsync(3, 3);
        await grain.SubtractAsync(5, 2);

        var addMetrics = grain.GetHandlerMetrics(CalculatorActorHandlerIds.Add);
        var subtractMetrics = grain.GetHandlerMetrics(CalculatorActorHandlerIds.Subtract);

        // Assert
        Assert.NotNull(addMetrics);
        Assert.Equal(3, addMetrics!.InvocationCount);
        Assert.Equal(3, addMetrics.SuccessCount);
        Assert.Equal(0, addMetrics.FailureCount);
        Assert.Equal(1.0, addMetrics.SuccessRate);

        Assert.NotNull(subtractMetrics);
        Assert.Equal(1, subtractMetrics!.InvocationCount);
    }

    /// <summary>
    /// Tests all handlers report metrics.
    /// </summary>
    [Fact]
    public async Task GetAllHandlerMetrics_ReturnsAllInvokedHandlers()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        await grain.AddAsync(1, 1);
        await grain.SubtractAsync(5, 2);
        await grain.FactorialAsync(5);

        var allMetrics = grain.GetAllHandlerMetrics();

        // Assert
        Assert.Equal(3, allMetrics.Count);
        Assert.Contains(CalculatorActorHandlerIds.Add, allMetrics.Keys);
        Assert.Contains(CalculatorActorHandlerIds.Subtract, allMetrics.Keys);
        Assert.Contains(CalculatorActorHandlerIds.Factorial, allMetrics.Keys);
    }

    /// <summary>
    /// Tests generated message struct sizes.
    /// </summary>
    [Fact]
    public void MessageStructs_HaveExpectedSizes()
    {
        // These sizes are critical for GPU memory layout
        unsafe
        {
            Assert.Equal(8, sizeof(AddRequest)); // 2 ints
            Assert.Equal(8, sizeof(AddResponse)); // 1 int + padding
            Assert.Equal(8, sizeof(SubtractRequest)); // 2 ints
            Assert.Equal(8, sizeof(SubtractResponse)); // 1 int + padding
            Assert.Equal(8, sizeof(MultiplyRequest)); // 2 ints
            Assert.Equal(8, sizeof(FactorialRequest)); // 1 int + padding
            Assert.Equal(8, sizeof(FactorialResponse)); // 1 long
            Assert.Equal(8, sizeof(GetAccumulatorResponse)); // 1 long
            Assert.Equal(32, sizeof(CalculatorActorState)); // State struct
        }
    }

    /// <summary>
    /// Tests handler IDs are sequential.
    /// </summary>
    [Fact]
    public void HandlerIds_AreSequentialAndCorrect()
    {
        Assert.Equal(1, CalculatorActorHandlerIds.Add);
        Assert.Equal(2, CalculatorActorHandlerIds.Subtract);
        Assert.Equal(3, CalculatorActorHandlerIds.Multiply);
        Assert.Equal(4, CalculatorActorHandlerIds.Factorial);
        Assert.Equal(5, CalculatorActorHandlerIds.GetAccumulator);
        Assert.Equal(6, CalculatorActorHandlerIds.Reset);
    }

    /// <summary>
    /// Tests kernel IDs are correctly formatted.
    /// </summary>
    [Fact]
    public void KernelIds_FollowNamingConvention()
    {
        Assert.Equal(
            "Orleans.GpuBridge.Grains.Generated.CalculatorActor.Dispatch",
            CalculatorActorKernels.DispatchKernelId);
    }

    /// <summary>
    /// Tests concurrent operations on the same grain.
    /// </summary>
    [Fact]
    public async Task ConcurrentOperations_MaintainConsistency()
    {
        // Arrange
        var grain = CreateCalculatorGrain();
        var tasks = new Task<int>[100];

        // Act - concurrent adds
        for (int i = 0; i < 100; i++)
        {
            tasks[i] = grain.AddAsync(1, i);
        }

        var results = await Task.WhenAll(tasks);

        // Assert - all results should be correct
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(1 + i, results[i]);
        }

        Assert.Equal(100, grain.TotalMessagesProcessed);
    }

    /// <summary>
    /// Tests large factorial computation.
    /// </summary>
    [Fact]
    public async Task FactorialAsync_HandlesLargeValues()
    {
        // Arrange
        var grain = CreateCalculatorGrain();

        // Act
        var result = await grain.FactorialAsync(20);

        // Assert
        Assert.Equal(2432902008176640000L, result);
    }

    private static CalculatorActorGrain CreateCalculatorGrain()
    {
        var grainContext = new TestGrainContext();
        var logger = NullLogger<CalculatorActorGrain>.Instance;
        var grain = new CalculatorActorGrain(grainContext, logger);

        // Simulate activation
        grain.OnActivateAsync(CancellationToken.None).GetAwaiter().GetResult();

        return grain;
    }
}

/// <summary>
/// Test implementation of IGrainContext for unit testing.
/// </summary>
internal sealed class TestGrainContext : IGrainContext, IEquatable<IGrainContext>
{
    private readonly GrainId _grainId;
    private readonly GrainAddress _address;
    private readonly ActivationId _activationId;

    public TestGrainContext()
    {
        _grainId = GrainId.Create("test", Guid.NewGuid().ToString());
        _activationId = ActivationId.NewId();
        _address = new GrainAddress
        {
            GrainId = _grainId,
            SiloAddress = SiloAddress.Zero
        };
    }

    public GrainReference GrainReference => throw new NotImplementedException();

    public GrainId GrainId => _grainId;

    public object? GrainInstance => null;

    public ActivationId ActivationId => _activationId;

    public GrainAddress Address => _address;

    public IServiceProvider ActivationServices => throw new NotImplementedException();

    public IGrainLifecycle ObservableLifecycle => throw new NotImplementedException();

    public IWorkItemScheduler Scheduler => throw new NotImplementedException();

    private readonly TaskCompletionSource _deactivatedTcs = new();

    /// <summary>
    /// Gets a task that completes when this activation has been deactivated.
    /// </summary>
    public Task Deactivated => _deactivatedTcs.Task;

    public void Activate(Dictionary<string, object>? requestContext, CancellationToken cancellationToken)
    {
    }

    public void Deactivate(DeactivationReason deactivationReason, CancellationToken cancellationToken)
    {
        _deactivatedTcs.TrySetResult();
    }

    public TTarget GetTarget<TTarget>() where TTarget : class
    {
        throw new NotImplementedException();
    }

    public TComponent? GetComponent<TComponent>() where TComponent : class
    {
        return default;
    }

    public void SetComponent<TComponent>(TComponent? instance) where TComponent : class
    {
    }

    public void Migrate(Dictionary<string, object>? requestContext, CancellationToken cancellationToken)
    {
    }

    public void ReceiveMessage(object message)
    {
    }

    public void Rehydrate(IRehydrationContext context)
    {
    }

    public bool Equals(IGrainContext? other)
    {
        if (other is null) return false;
        return _grainId.Equals(other.GrainId) && _activationId.Equals(other.ActivationId);
    }

    public override bool Equals(object? obj)
    {
        return obj is IGrainContext context && Equals(context);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(_grainId, _activationId);
    }
}
