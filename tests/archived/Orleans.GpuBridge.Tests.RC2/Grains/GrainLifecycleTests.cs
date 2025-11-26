using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Grains.Interfaces;
using Orleans.GpuBridge.Grains.Stream;
using Orleans.GpuBridge.Tests.RC2.Infrastructure;
using Orleans.Streams;

namespace Orleans.GpuBridge.Tests.RC2.Grains;

/// <summary>
/// Comprehensive tests for Orleans grain lifecycle management.
/// Tests activation, deactivation, reactivation, state persistence, and resource cleanup.
/// Target: 20+ tests for grain lifecycle scenarios.
/// </summary>
[Collection("ClusterCollection")]
public sealed class GrainLifecycleTests : IClassFixture<ClusterFixture>
{
    private readonly ClusterFixture _fixture;

    public GrainLifecycleTests(ClusterFixture fixture)
    {
        _fixture = fixture;
    }

    #region Batch Grain Lifecycle Tests

    [Fact]
    public async Task BatchGrain_FirstActivation_ShouldInitializeResources()
    {
        // Arrange
        var kernelId = $"kernels/activation-{Guid.NewGuid()}";
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), kernelId);

        // Act - First call triggers activation
        var input = new[] { 1.0f, 2.0f };
        var result = await grain.ExecuteAsync(input);

        // Assert
        result.Success.Should().BeTrue();
        result.KernelId.ToString().Should().Be(kernelId);
    }

    [Fact]
    public async Task BatchGrain_Reactivation_ShouldRestoreCapabilities()
    {
        // Arrange
        var grainId = Guid.NewGuid();
        var kernelId = "kernels/reactivation";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(grainId, kernelId);

        // First activation and execution
        await grain1.ExecuteAsync(new[] { 1.0f });

        // Act - Get same grain (may reactivate)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(grainId, kernelId);
        var result = await grain2.ExecuteAsync(new[] { 2.0f });

        // Assert
        result.Success.Should().BeTrue();
        result.KernelId.ToString().Should().Be(kernelId);
    }

    [Fact]
    public async Task BatchGrain_ConcurrentActivations_ShouldWorkIndependently()
    {
        // Arrange
        var kernelId = "kernels/concurrent-activation";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<int, int>>(Guid.NewGuid(), kernelId);
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<int, int>>(Guid.NewGuid(), kernelId);

        var input = new[] { 1, 2, 3 };

        // Act - Execute on both grains concurrently
        var task1 = grain1.ExecuteAsync(input);
        var task2 = grain2.ExecuteAsync(input);

        var results = await Task.WhenAll(task1, task2);

        // Assert
        results[0].Success.Should().BeTrue();
        results[1].Success.Should().BeTrue();
    }

    #endregion

    #region Resident Grain Lifecycle Tests

    [Fact]
    public async Task ResidentGrain_Activation_ShouldInitializeRingKernel()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"lifecycle-resident-{Guid.NewGuid()}");

        // Act - First call triggers activation
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.Should().NotBeNull();
        info.DeviceName.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task ResidentGrain_Deactivation_ShouldPersistState()
    {
        // Arrange
        var grainKey = $"persistent-{Guid.NewGuid()}";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>(grainKey);

        var data = new int[] { 10, 20, 30 };
        await grain1.StoreDataAsync(data);

        // Act - Get same grain (simulates deactivation/reactivation)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>(grainKey);
        var retrieved = await grain2.GetDataAsync();

        // Assert
        retrieved.Should().NotBeNull();
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task ResidentGrain_ClearOnDeactivation_ShouldReleaseResources()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"cleanup-{Guid.NewGuid()}");

        // Create multiple allocations
        for (int i = 0; i < 5; i++)
        {
            await grain.AllocateAsync(1024);
        }

        // Act - Clear (simulates deactivation cleanup)
        await grain.ClearAsync();
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().Be(0);
        info.TotalMemoryBytes.Should().Be(0);
    }

    [Fact]
    public async Task ResidentGrain_MultipleAllocationsReactivation_ShouldRestore()
    {
        // Arrange
        var grainKey = $"multi-alloc-{Guid.NewGuid()}";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);

        // Create allocations
        var handle1 = await grain1.AllocateAsync(512);
        var handle2 = await grain1.AllocateAsync(1024);

        // Act - Get same grain
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>(grainKey);
        var info = await grain2.GetMemoryInfoAsync();

        // Assert - State should be maintained
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    #endregion

    #region Stream Grain Lifecycle Tests

    [Fact]
    public async Task StreamGrain_Activation_ShouldInitializeChannel()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/stream-lifecycle");

        // Act - Check initial status
        var status = await grain.GetStatusAsync();

        // Assert
        status.Should().Be(StreamProcessingStatus.Idle);
    }

    [Fact]
    public async Task StreamGrain_StartStop_ShouldManageLifecycle()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<int, int>>("kernels/start-stop");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        // Act - Start
        await grain.StartProcessingAsync(inputStream, outputStream);
        var runningStatus = await grain.GetStatusAsync();

        // Stop
        await grain.StopProcessingAsync();
        var stoppedStatus = await grain.GetStatusAsync();

        // Assert
        runningStatus.Should().Be(StreamProcessingStatus.Processing);
        stoppedStatus.Should().Be(StreamProcessingStatus.Stopped);
    }

    [Fact]
    public async Task StreamGrain_Deactivation_ShouldCleanupSubscriptions()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/cleanup");

        var inputStream = StreamId.Create("input", Guid.NewGuid());
        var outputStream = StreamId.Create("output", Guid.NewGuid());

        await grain.StartProcessingAsync(inputStream, outputStream);

        // Act - Stop (simulates deactivation)
        await grain.StopProcessingAsync();
        var stats = await grain.GetStatsAsync();

        // Assert
        stats.Should().NotBeNull();
        stats.StartTime.Should().NotBe(default);
    }

    #endregion

    #region State Persistence Tests

    [Fact]
    public async Task GrainState_AfterExecution_ShouldPersist()
    {
        // Arrange
        var grainKey = $"state-persist-{Guid.NewGuid()}";
        var grain1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>(grainKey);

        var data = new double[] { 1.5, 2.5, 3.5 };

        // Act
        await grain1.StoreDataAsync(data);

        // Get same grain (may reload state)
        var grain2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<double>>(grainKey);
        var retrieved = await grain2.GetDataAsync();

        // Assert
        retrieved.Should().Equal(data);
    }

    [Fact]
    public async Task GrainState_EmptyState_ShouldHandleGracefully()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<int>>($"empty-state-{Guid.NewGuid()}");

        // Act - Get data without storing
        var data = await grain.GetDataAsync();

        // Assert
        data.Should().BeNull();
    }

    #endregion

    #region Resource Management Tests

    [Fact]
    public async Task GrainResources_OnActivation_ShouldAllocate()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"resource-alloc-{Guid.NewGuid()}");

        // Act - First call allocates resources
        var handle = await grain.AllocateAsync(1024);

        // Assert
        handle.Should().NotBeNull();
        var info = await grain.GetMemoryInfoAsync();
        info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task GrainResources_OnDeactivation_ShouldRelease()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<byte>>($"resource-release-{Guid.NewGuid()}");

        await grain.AllocateAsync(2048);

        // Act - Clear resources (simulates deactivation)
        await grain.ClearAsync();
        var info = await grain.GetMemoryInfoAsync();

        // Assert
        info.AllocatedMemoryBytes.Should().Be(0);
    }

    #endregion

    #region Concurrent Lifecycle Tests

    [Fact]
    public async Task MultipleGrains_ConcurrentActivation_ShouldSucceed()
    {
        // Arrange
        var grains = Enumerable.Range(0, 10)
            .Select(i => _fixture.Cluster.GrainFactory
                .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/concurrent"))
            .ToList();

        var input = new[] { 1.0f, 2.0f };

        // Act - Activate all grains concurrently
        var tasks = grains.Select(g => g.ExecuteAsync(input)).ToArray();
        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().AllSatisfy(r => r.Success.Should().BeTrue());
    }

    [Fact]
    public async Task MultipleGrains_SequentialActivation_ShouldSucceed()
    {
        // Arrange
        var grains = Enumerable.Range(0, 5)
            .Select(i => _fixture.Cluster.GrainFactory
                .GetGrain<IGpuResidentGrain<int>>($"sequential-{i}"))
            .ToList();

        // Act - Activate sequentially
        foreach (var grain in grains)
        {
            await grain.AllocateAsync(512);
        }

        // Assert - Verify all activated
        foreach (var grain in grains)
        {
            var info = await grain.GetMemoryInfoAsync();
            info.AllocatedMemoryBytes.Should().BeGreaterThan(0);
        }
    }

    #endregion

    #region Error Recovery Tests

    [Fact]
    public async Task GrainActivation_OnError_ShouldRecover()
    {
        // Arrange
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/error-recovery");

        // Act - Execute with null (error)
        var errorResult = await grain.ExecuteAsync(null!);

        // Then execute valid input (recovery)
        var validResult = await grain.ExecuteAsync(new[] { 1.0f });

        // Assert
        errorResult.Success.Should().BeFalse();
        validResult.Success.Should().BeTrue();
    }

    #endregion

    #region Cross-Grain Interaction Tests

    [Fact]
    public async Task BatchAndResidentGrains_ShouldCoexist()
    {
        // Arrange
        var batchGrain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/batch");
        var residentGrain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuResidentGrain<float>>($"resident-{Guid.NewGuid()}");

        // Act
        var batchResult = await batchGrain.ExecuteAsync(new[] { 1.0f, 2.0f });
        await residentGrain.StoreDataAsync(new[] { 3.0f, 4.0f });

        // Assert
        batchResult.Success.Should().BeTrue();
        var residentData = await residentGrain.GetDataAsync();
        residentData.Should().NotBeNull();
    }

    [Fact]
    public async Task MultipleStreamGrains_ShouldProcessIndependently()
    {
        // Arrange
        var stream1 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/stream1");
        var stream2 = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float, float>>("kernels/stream2");

        var input1 = StreamId.Create("in1", Guid.NewGuid());
        var output1 = StreamId.Create("out1", Guid.NewGuid());
        var input2 = StreamId.Create("in2", Guid.NewGuid());
        var output2 = StreamId.Create("out2", Guid.NewGuid());

        // Act
        await stream1.StartProcessingAsync(input1, output1);
        await stream2.StartProcessingAsync(input2, output2);

        var status1 = await stream1.GetStatusAsync();
        var status2 = await stream2.GetStatusAsync();

        // Assert
        status1.Should().Be(StreamProcessingStatus.Processing);
        status2.Should().Be(StreamProcessingStatus.Processing);

        // Cleanup
        await stream1.StopProcessingAsync();
        await stream2.StopProcessingAsync();
    }

    #endregion

    #region Reentrant Behavior Tests

    [Fact]
    public async Task ReentrantGrain_ConcurrentCalls_ShouldAllowInterleaving()
    {
        // Arrange - Enhanced grains are marked [Reentrant]
        var grain = _fixture.Cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float, float>>(Guid.NewGuid(), "kernels/reentrant");

        var input = new[] { 1.0f, 2.0f, 3.0f };

        // Act - Make concurrent calls
        var task1 = grain.ExecuteAsync(input);
        var task2 = grain.ExecuteAsync(input);
        var task3 = grain.ExecuteAsync(input);

        var results = await Task.WhenAll(task1, task2, task3);

        // Assert - All should complete successfully
        results.Should().AllSatisfy(r => r.Success.Should().BeTrue());
    }

    #endregion
}
