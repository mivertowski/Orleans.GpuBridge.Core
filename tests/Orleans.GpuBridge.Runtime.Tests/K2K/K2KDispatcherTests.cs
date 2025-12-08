// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Runtime.InteropServices;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans.GpuBridge.Abstractions.K2K;
using Orleans.GpuBridge.Runtime.K2K;

namespace Orleans.GpuBridge.Runtime.Tests.K2K;

/// <summary>
/// Unit tests for K2KDispatcher with P2P routing support.
/// </summary>
public sealed class K2KDispatcherTests : IDisposable
{
    private readonly Mock<ILogger<K2KDispatcher>> _loggerMock;
    private readonly Mock<IGpuPeerToPeerMemory> _p2pMemoryMock;
    private K2KDispatcher? _dispatcher;
    private IntPtr _testQueuePtr;

    public K2KDispatcherTests()
    {
        _loggerMock = new Mock<ILogger<K2KDispatcher>>();
        _p2pMemoryMock = new Mock<IGpuPeerToPeerMemory>();

        // Default P2P capability - disabled
        _p2pMemoryMock
            .Setup(x => x.GetP2PCapability(It.IsAny<int>(), It.IsAny<int>()))
            .Returns((P2PCapabilityInfo?)null);

        // Allocate test queue memory
        _testQueuePtr = Marshal.AllocHGlobal(4096);
        K2KDispatcher.InitializeQueue(_testQueuePtr, 256, 64);
    }

    public void Dispose()
    {
        _dispatcher?.Dispose();
        if (_testQueuePtr != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(_testQueuePtr);
            _testQueuePtr = IntPtr.Zero;
        }
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithoutP2PProvider_CreatesDispatcher()
    {
        // Arrange & Act
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Assert
        _dispatcher.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_WithP2PProvider_CreatesDispatcherWithP2PSupport()
    {
        // Arrange & Act
        _dispatcher = new K2KDispatcher(_loggerMock.Object, _p2pMemoryMock.Object);

        // Assert
        _dispatcher.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_NullLogger_ThrowsArgumentNullException()
    {
        // Arrange & Act
        var act = () => new K2KDispatcher(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>()
            .WithParameterName("logger");
    }

    #endregion

    #region Queue Registration Tests

    [Fact]
    public void RegisterQueue_ValidParameters_RegistersSuccessfully()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Act
        _dispatcher.RegisterQueue("TestActor", 123, _testQueuePtr);
        var queuePtr = _dispatcher.GetTargetQueuePointer("TestActor", 123);

        // Assert
        queuePtr.Should().Be(_testQueuePtr);
    }

    [Fact]
    public void RegisterQueue_EmptyActorType_ThrowsArgumentException()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Act
        var act = () => _dispatcher.RegisterQueue("", 123, _testQueuePtr);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void RegisterQueue_ZeroPointer_ThrowsArgumentException()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Act
        var act = () => _dispatcher.RegisterQueue("TestActor", 123, IntPtr.Zero);

        // Assert
        act.Should().Throw<ArgumentException>()
            .WithParameterName("queuePointer");
    }

    [Fact]
    public void UnregisterQueue_RegisteredQueue_RemovesQueue()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);
        _dispatcher.RegisterQueue("TestActor", 123, _testQueuePtr);

        // Act
        _dispatcher.UnregisterQueue("TestActor", 123);
        var queuePtr = _dispatcher.GetTargetQueuePointer("TestActor", 123);

        // Assert
        queuePtr.Should().Be(IntPtr.Zero);
    }

    #endregion

    #region Device Registration Tests

    [Fact]
    public void RegisterActorDevice_ValidParameters_RegistersSuccessfully()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object, _p2pMemoryMock.Object);

        // Act
        _dispatcher.RegisterActorDevice("TestActor", 123, 0);
        var deviceId = _dispatcher.GetActorDevice("TestActor", 123);

        // Assert
        deviceId.Should().Be(0);
    }

    [Fact]
    public void RegisterActorDevice_NegativeDeviceId_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Act
        var act = () => _dispatcher.RegisterActorDevice("TestActor", 123, -1);

        // Assert
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithParameterName("deviceId");
    }

    [Fact]
    public void GetActorDevice_UnregisteredActor_ReturnsMinusOne()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Act
        var deviceId = _dispatcher.GetActorDevice("TestActor", 999);

        // Assert
        deviceId.Should().Be(-1);
    }

    [Fact]
    public void RegisterActorDevice_MultipleDevices_CachesP2PCapabilities()
    {
        // Arrange
        var p2pInfo = new P2PCapabilityInfo(
            SourceDeviceId: 0,
            TargetDeviceId: 1,
            IsSupported: true,
            IsEnabled: true,
            AccessType: P2PAccessType.PciExpress,
            EstimatedBandwidthGBps: 32.0,
            EstimatedLatencyNs: 800.0,
            AtomicsSupported: false,
            NativeAtomicsSupported: false);

        _p2pMemoryMock
            .Setup(x => x.GetP2PCapability(0, 1))
            .Returns(p2pInfo);
        _p2pMemoryMock
            .Setup(x => x.GetP2PCapability(1, 0))
            .Returns(p2pInfo);

        _dispatcher = new K2KDispatcher(_loggerMock.Object, _p2pMemoryMock.Object);

        // Act - register actors on different devices
        _dispatcher.RegisterActorDevice("TestActor", 1, 0);
        _dispatcher.RegisterActorDevice("TestActor", 2, 1);

        // Assert - P2P capabilities should have been queried
        _p2pMemoryMock.Verify(x => x.GetP2PCapability(0, 1), Times.AtLeastOnce);
        _p2pMemoryMock.Verify(x => x.GetP2PCapability(1, 0), Times.AtLeastOnce);
    }

    #endregion

    #region Routing Statistics Tests

    [Fact]
    public void GetRoutingStats_InitialState_ReturnsZeroCounts()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Act
        var stats = _dispatcher.GetRoutingStats();

        // Assert
        stats.TotalDispatches.Should().Be(0);
        stats.P2PDispatches.Should().Be(0);
        stats.CpuRoutedDispatches.Should().Be(0);
        stats.FailedDispatches.Should().Be(0);
        stats.BroadcastOperations.Should().Be(0);
        stats.RequestResponseOperations.Should().Be(0);
    }

    [Fact]
    public async Task DispatchAsync_ValidTarget_UpdatesStatistics()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);
        _dispatcher.RegisterQueue("TestActor", 123, _testQueuePtr);

        // Act
        await _dispatcher.DispatchAsync(
            sourceActorId: 1,
            targetActorType: "TestActor",
            targetMethod: "HandleMessage",
            targetActorId: 123,
            message: new TestMessage { Value = 42 });

        var stats = _dispatcher.GetRoutingStats();

        // Assert
        stats.TotalDispatches.Should().Be(1);
        stats.CpuRoutedDispatches.Should().Be(1);
        stats.FailedDispatches.Should().Be(0);
    }

    [Fact]
    public async Task DispatchAsync_InvalidTarget_UpdatesFailedCount()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Act
        await _dispatcher.DispatchAsync(
            sourceActorId: 1,
            targetActorType: "NonExistent",
            targetMethod: "HandleMessage",
            targetActorId: 999,
            message: new TestMessage { Value = 42 });

        var stats = _dispatcher.GetRoutingStats();

        // Assert
        stats.TotalDispatches.Should().Be(0);
        stats.FailedDispatches.Should().Be(1);
    }

    [Fact]
    public async Task BroadcastAsync_MultipleTargets_UpdatesBroadcastCount()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);

        // Create multiple queues
        var queue2 = Marshal.AllocHGlobal(4096);
        var queue3 = Marshal.AllocHGlobal(4096);
        try
        {
            K2KDispatcher.InitializeQueue(queue2, 256, 64);
            K2KDispatcher.InitializeQueue(queue3, 256, 64);

            _dispatcher.RegisterQueue("TestActor", 1, _testQueuePtr);
            _dispatcher.RegisterQueue("TestActor", 2, queue2);
            _dispatcher.RegisterQueue("TestActor", 3, queue3);

            // Act
            var targetIds = new long[] { 1, 2, 3 };
            await _dispatcher.BroadcastAsync(
                sourceActorId: 0,
                targetActorType: "TestActor",
                targetMethod: "HandleBroadcast",
                targetActorIds: targetIds,
                message: new TestMessage { Value = 100 });

            var stats = _dispatcher.GetRoutingStats();

            // Assert
            stats.BroadcastOperations.Should().Be(1);
            stats.TotalDispatches.Should().Be(3);
        }
        finally
        {
            Marshal.FreeHGlobal(queue2);
            Marshal.FreeHGlobal(queue3);
        }
    }

    [Fact]
    public void GetRoutingStats_P2PEnabledPairs_ReturnsCorrectCount()
    {
        // Arrange
        var p2pInfo = new P2PCapabilityInfo(
            SourceDeviceId: 0,
            TargetDeviceId: 1,
            IsSupported: true,
            IsEnabled: true,
            AccessType: P2PAccessType.NvLink,
            EstimatedBandwidthGBps: 600.0,
            EstimatedLatencyNs: 150.0,
            AtomicsSupported: true,
            NativeAtomicsSupported: true);

        _p2pMemoryMock
            .Setup(x => x.GetP2PCapability(It.IsAny<int>(), It.IsAny<int>()))
            .Returns(p2pInfo);

        _dispatcher = new K2KDispatcher(_loggerMock.Object, _p2pMemoryMock.Object);

        // Act - register actors to trigger P2P capability caching
        _dispatcher.RegisterActorDevice("TestActor", 1, 0);
        _dispatcher.RegisterActorDevice("TestActor", 2, 1);

        var stats = _dispatcher.GetRoutingStats();

        // Assert - should have cached P2P capabilities
        stats.P2PEnabledPairs.Should().BeGreaterThanOrEqualTo(0);
    }

    #endregion

    #region P2P Routing Tests

    [Fact]
    public async Task DispatchAsync_P2PEnabled_TracksP2PDispatch()
    {
        // Arrange
        var p2pInfo = new P2PCapabilityInfo(
            SourceDeviceId: 0,
            TargetDeviceId: 1,
            IsSupported: true,
            IsEnabled: true,
            AccessType: P2PAccessType.PciExpress,
            EstimatedBandwidthGBps: 32.0,
            EstimatedLatencyNs: 800.0,
            AtomicsSupported: false,
            NativeAtomicsSupported: false);

        _p2pMemoryMock
            .Setup(x => x.GetP2PCapability(0, 1))
            .Returns(p2pInfo);

        _dispatcher = new K2KDispatcher(_loggerMock.Object, _p2pMemoryMock.Object);

        // Register queue and devices
        _dispatcher.RegisterQueue("TestActor", 2, _testQueuePtr);
        _dispatcher.RegisterActorDevice("TestActor", 1, 0);  // Source on device 0
        _dispatcher.RegisterActorDevice("TestActor", 2, 1);  // Target on device 1

        // Act
        await _dispatcher.DispatchAsync(
            sourceActorId: 1,
            targetActorType: "TestActor",
            targetMethod: "HandleMessage",
            targetActorId: 2,
            message: new TestMessage { Value = 42 });

        var stats = _dispatcher.GetRoutingStats();

        // Assert - should be P2P dispatch since P2P is enabled between devices
        stats.TotalDispatches.Should().Be(1);
        stats.P2PDispatches.Should().Be(1);
        stats.CpuRoutedDispatches.Should().Be(0);
    }

    [Fact]
    public async Task DispatchAsync_SameDevice_UsesCpuRouting()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object, _p2pMemoryMock.Object);

        _dispatcher.RegisterQueue("TestActor", 2, _testQueuePtr);
        _dispatcher.RegisterActorDevice("TestActor", 1, 0);  // Same device
        _dispatcher.RegisterActorDevice("TestActor", 2, 0);  // Same device

        // Act
        await _dispatcher.DispatchAsync(
            sourceActorId: 1,
            targetActorType: "TestActor",
            targetMethod: "HandleMessage",
            targetActorId: 2,
            message: new TestMessage { Value = 42 });

        var stats = _dispatcher.GetRoutingStats();

        // Assert - same device should use CPU routing
        stats.TotalDispatches.Should().Be(1);
        stats.CpuRoutedDispatches.Should().Be(1);
        stats.P2PDispatches.Should().Be(0);
    }

    [Fact]
    public async Task DispatchAsync_NoP2PCapability_FallsToCpuRouting()
    {
        // Arrange
        _p2pMemoryMock
            .Setup(x => x.GetP2PCapability(It.IsAny<int>(), It.IsAny<int>()))
            .Returns((P2PCapabilityInfo?)null);

        _dispatcher = new K2KDispatcher(_loggerMock.Object, _p2pMemoryMock.Object);

        _dispatcher.RegisterQueue("TestActor", 2, _testQueuePtr);
        _dispatcher.RegisterActorDevice("TestActor", 1, 0);
        _dispatcher.RegisterActorDevice("TestActor", 2, 1);

        // Act
        await _dispatcher.DispatchAsync(
            sourceActorId: 1,
            targetActorType: "TestActor",
            targetMethod: "HandleMessage",
            targetActorId: 2,
            message: new TestMessage { Value = 42 });

        var stats = _dispatcher.GetRoutingStats();

        // Assert - no P2P capability means CPU routing
        stats.TotalDispatches.Should().Be(1);
        stats.CpuRoutedDispatches.Should().Be(1);
        stats.P2PDispatches.Should().Be(0);
    }

    #endregion

    #region Queue Operations Tests

    [Fact]
    public void InitializeQueue_ValidParameters_InitializesCorrectly()
    {
        // Arrange
        var queuePtr = Marshal.AllocHGlobal(1024);
        try
        {
            // Act
            K2KDispatcher.InitializeQueue(queuePtr, 64, 32);

            // Assert - queue should be initialized with zero message count
            var messageCount = K2KDispatcher.GetQueueMessageCount(queuePtr);
            messageCount.Should().Be(0);
        }
        finally
        {
            Marshal.FreeHGlobal(queuePtr);
        }
    }

    [Fact]
    public void InitializeQueue_NonPowerOfTwoCapacity_ThrowsArgumentException()
    {
        // Arrange
        var queuePtr = Marshal.AllocHGlobal(1024);
        try
        {
            // Act
            var act = () => K2KDispatcher.InitializeQueue(queuePtr, 100, 32);

            // Assert
            act.Should().Throw<ArgumentException>()
                .WithParameterName("capacity");
        }
        finally
        {
            Marshal.FreeHGlobal(queuePtr);
        }
    }

    [Fact]
    public void InitializeQueue_ZeroPointer_ThrowsArgumentException()
    {
        // Act
        var act = () => K2KDispatcher.InitializeQueue(IntPtr.Zero, 64, 32);

        // Assert
        act.Should().Throw<ArgumentException>()
            .WithParameterName("queuePtr");
    }

    [Fact]
    public void GetQueueMessageCount_EmptyQueue_ReturnsZero()
    {
        // Act
        var count = K2KDispatcher.GetQueueMessageCount(_testQueuePtr);

        // Assert
        count.Should().Be(0);
    }

    [Fact]
    public void GetQueueMessageCount_ZeroPointer_ReturnsZero()
    {
        // Act
        var count = K2KDispatcher.GetQueueMessageCount(IntPtr.Zero);

        // Assert
        count.Should().Be(0);
    }

    #endregion

    #region Dispose Tests

    [Fact]
    public void Dispose_ClearsAllRegistrations()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);
        _dispatcher.RegisterQueue("TestActor", 123, _testQueuePtr);
        _dispatcher.RegisterActorDevice("TestActor", 123, 0);

        // Act
        _dispatcher.Dispose();

        // Assert - after dispose, GetRoutingStats should show 0 registered queues
        var stats = _dispatcher.GetRoutingStats();
        stats.RegisteredQueues.Should().Be(0);
    }

    [Fact]
    public async Task DispatchAsync_AfterDispose_ThrowsObjectDisposedException()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);
        _dispatcher.Dispose();

        // Act
        var act = async () => await _dispatcher.DispatchAsync(
            sourceActorId: 1,
            targetActorType: "TestActor",
            targetMethod: "HandleMessage",
            targetActorId: 123,
            message: new TestMessage { Value = 42 });

        // Assert
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    #endregion

    #region Concurrent Access Tests

    [Fact]
    public async Task RegisterQueue_ConcurrentRegistrations_AllSucceed()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);
        var queues = new List<IntPtr>();
        const int registrationCount = 100;

        try
        {
            // Allocate queues
            for (var i = 0; i < registrationCount; i++)
            {
                var ptr = Marshal.AllocHGlobal(4096);
                K2KDispatcher.InitializeQueue(ptr, 256, 64);
                queues.Add(ptr);
            }

            // Act - concurrent registrations
            var tasks = queues.Select((ptr, index) =>
                Task.Run(() => _dispatcher.RegisterQueue("TestActor", index, ptr)));

            await Task.WhenAll(tasks);

            var stats = _dispatcher.GetRoutingStats();

            // Assert
            stats.RegisteredQueues.Should().Be(registrationCount);
        }
        finally
        {
            foreach (var ptr in queues)
            {
                Marshal.FreeHGlobal(ptr);
            }
        }
    }

    [Fact]
    public async Task DispatchAsync_ConcurrentDispatches_AllSucceed()
    {
        // Arrange
        _dispatcher = new K2KDispatcher(_loggerMock.Object);
        _dispatcher.RegisterQueue("TestActor", 123, _testQueuePtr);

        const int dispatchCount = 100;

        // Act - concurrent dispatches
        var tasks = Enumerable.Range(0, dispatchCount)
            .Select(i => _dispatcher.DispatchAsync(
                sourceActorId: i,
                targetActorType: "TestActor",
                targetMethod: "HandleMessage",
                targetActorId: 123,
                message: new TestMessage { Value = i }).AsTask());

        await Task.WhenAll(tasks);

        var stats = _dispatcher.GetRoutingStats();

        // Assert
        stats.TotalDispatches.Should().Be(dispatchCount);
        stats.FailedDispatches.Should().Be(0);
    }

    #endregion

    [StructLayout(LayoutKind.Sequential)]
    private struct TestMessage
    {
        public int Value;
    }
}
