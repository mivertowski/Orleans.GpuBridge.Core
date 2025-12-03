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
/// Unit tests for CpuFallbackPeerToPeerMemory.
/// </summary>
public sealed class CpuFallbackPeerToPeerMemoryTests : IDisposable
{
    private readonly Mock<ILogger<CpuFallbackPeerToPeerMemory>> _loggerMock;
    private readonly CpuFallbackPeerToPeerMemory _provider;
    private IntPtr _sourcePtr;
    private IntPtr _destPtr;
    private const int BufferSize = 256;

    public CpuFallbackPeerToPeerMemoryTests()
    {
        _loggerMock = new Mock<ILogger<CpuFallbackPeerToPeerMemory>>();
        _provider = new CpuFallbackPeerToPeerMemory(_loggerMock.Object);

        // Allocate test buffers
        _sourcePtr = Marshal.AllocHGlobal(BufferSize);
        _destPtr = Marshal.AllocHGlobal(BufferSize);

        // Initialize source with test data
        unsafe
        {
            var sourceSpan = new Span<byte>((void*)_sourcePtr, BufferSize);
            for (var i = 0; i < BufferSize; i++)
            {
                sourceSpan[i] = (byte)(i % 256);
            }
        }
    }

    public void Dispose()
    {
        if (_sourcePtr != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(_sourcePtr);
            _sourcePtr = IntPtr.Zero;
        }
        if (_destPtr != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(_destPtr);
            _destPtr = IntPtr.Zero;
        }
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_NullLogger_ThrowsArgumentNullException()
    {
        // Act
        var act = () => new CpuFallbackPeerToPeerMemory(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>()
            .WithParameterName("logger");
    }

    [Fact]
    public void Constructor_ValidLogger_CreatesInstance()
    {
        // Act & Assert
        _provider.Should().NotBeNull();
    }

    #endregion

    #region CanAccessPeer Tests

    [Fact]
    public void CanAccessPeer_AnyDevices_ReturnsFalse()
    {
        // Act
        var result = _provider.CanAccessPeer(0, 1);

        // Assert - CPU fallback never reports true P2P capability
        result.Should().BeFalse();
    }

    [Fact]
    public void CanAccessPeer_SameDevice_ReturnsFalse()
    {
        // Act
        var result = _provider.CanAccessPeer(0, 0);

        // Assert
        result.Should().BeFalse();
    }

    #endregion

    #region EnablePeerAccess Tests

    [Fact]
    public async Task EnablePeerAccessAsync_AnyDevices_ReturnsTrue()
    {
        // Act
        var result = await _provider.EnablePeerAccessAsync(0, 1);

        // Assert - CPU fallback always succeeds (it's a no-op)
        result.Should().BeTrue();
    }

    [Fact]
    public async Task EnablePeerAccessAsync_WithCancellationToken_Completes()
    {
        // Arrange
        using var cts = new CancellationTokenSource();

        // Act
        var result = await _provider.EnablePeerAccessAsync(0, 1, cts.Token);

        // Assert
        result.Should().BeTrue();
    }

    #endregion

    #region DisablePeerAccess Tests

    [Fact]
    public async Task DisablePeerAccessAsync_AnyDevices_Completes()
    {
        // Act
        await _provider.DisablePeerAccessAsync(0, 1);

        // Assert - no exception means success
    }

    #endregion

    #region CopyPeerToPeer Tests

    [Fact]
    public void CopyPeerToPeer_ValidParameters_CopiesData()
    {
        // Act
        _provider.CopyPeerToPeer(_sourcePtr, 0, _destPtr, 1, BufferSize);

        // Assert - verify data was copied
        unsafe
        {
            var sourceSpan = new Span<byte>((void*)_sourcePtr, BufferSize);
            var destSpan = new Span<byte>((void*)_destPtr, BufferSize);

            for (var i = 0; i < BufferSize; i++)
            {
                destSpan[i].Should().Be(sourceSpan[i], $"byte at position {i}");
            }
        }
    }

    [Fact]
    public void CopyPeerToPeer_ZeroSize_DoesNothing()
    {
        // Arrange - zero out destination first
        unsafe
        {
            var destSpan = new Span<byte>((void*)_destPtr, BufferSize);
            destSpan.Clear();
        }

        // Act
        _provider.CopyPeerToPeer(_sourcePtr, 0, _destPtr, 1, 0);

        // Assert - destination should still be zeros
        unsafe
        {
            var destSpan = new Span<byte>((void*)_destPtr, BufferSize);
            for (var i = 0; i < BufferSize; i++)
            {
                destSpan[i].Should().Be(0);
            }
        }
    }

    [Fact]
    public void CopyPeerToPeer_NullSourcePointer_ThrowsArgumentException()
    {
        // Act
        var act = () => _provider.CopyPeerToPeer(IntPtr.Zero, 0, _destPtr, 1, BufferSize);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CopyPeerToPeer_NullDestinationPointer_ThrowsArgumentException()
    {
        // Act
        var act = () => _provider.CopyPeerToPeer(_sourcePtr, 0, IntPtr.Zero, 1, BufferSize);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CopyPeerToPeer_NegativeSize_DoesNothing()
    {
        // Arrange - zero out destination
        unsafe
        {
            var destSpan = new Span<byte>((void*)_destPtr, BufferSize);
            destSpan.Clear();
        }

        // Act - negative size should be treated as nothing to copy
        _provider.CopyPeerToPeer(_sourcePtr, 0, _destPtr, 1, -1);

        // Assert - destination should still be zeros (no copy happened)
        unsafe
        {
            var destSpan = new Span<byte>((void*)_destPtr, BufferSize);
            for (var i = 0; i < BufferSize; i++)
            {
                destSpan[i].Should().Be(0);
            }
        }
    }

    [Fact]
    public async Task CopyPeerToPeerAsync_ValidParameters_CopiesData()
    {
        // Act
        await _provider.CopyPeerToPeerAsync(_sourcePtr, 0, _destPtr, 1, BufferSize);

        // Assert - verify data was copied
        unsafe
        {
            var sourceSpan = new Span<byte>((void*)_sourcePtr, BufferSize);
            var destSpan = new Span<byte>((void*)_destPtr, BufferSize);

            for (var i = 0; i < BufferSize; i++)
            {
                destSpan[i].Should().Be(sourceSpan[i]);
            }
        }
    }

    #endregion

    #region MapPeerMemory Tests

    [Fact]
    public async Task MapPeerMemoryAsync_ValidParameters_ReturnsSamePointer()
    {
        // Act
        var mappedPtr = await _provider.MapPeerMemoryAsync(_sourcePtr, 0, 1, BufferSize);

        // Assert - CPU fallback returns same pointer (unified memory assumption)
        mappedPtr.Should().Be(_sourcePtr);
    }

    [Fact]
    public async Task UnmapPeerMemoryAsync_AnyPointer_Completes()
    {
        // Arrange
        var mappedPtr = await _provider.MapPeerMemoryAsync(_sourcePtr, 0, 1, BufferSize);

        // Act
        await _provider.UnmapPeerMemoryAsync(mappedPtr, 1);

        // Assert - no exception means success
    }

    #endregion

    #region GetP2PCapability Tests

    [Fact]
    public void GetP2PCapability_AnyDevices_ReturnsCapabilityInfo()
    {
        // Act
        var capability = _provider.GetP2PCapability(0, 1);

        // Assert
        capability.Should().NotBeNull();
        capability!.SourceDeviceId.Should().Be(0);
        capability.TargetDeviceId.Should().Be(1);
    }

    [Fact]
    public void GetP2PCapability_ReturnsNotSupported()
    {
        // Act
        var capability = _provider.GetP2PCapability(0, 1);

        // Assert - CPU fallback reports no true P2P
        capability!.IsSupported.Should().BeFalse();
        capability.AccessType.Should().Be(P2PAccessType.None);
    }

    [Fact]
    public void GetP2PCapability_ReturnsEnabled()
    {
        // Act
        var capability = _provider.GetP2PCapability(0, 1);

        // Assert - CPU routing is always "enabled"
        capability!.IsEnabled.Should().BeTrue();
    }

    [Fact]
    public void GetP2PCapability_ReturnsReasonableBandwidth()
    {
        // Act
        var capability = _provider.GetP2PCapability(0, 1);

        // Assert - should report typical PCIe bandwidth
        capability!.EstimatedBandwidthGBps.Should().BeInRange(10.0, 30.0);
    }

    [Fact]
    public void GetP2PCapability_ReturnsReasonableLatency()
    {
        // Act
        var capability = _provider.GetP2PCapability(0, 1);

        // Assert - should report CPU staging latency (~5μs)
        capability!.EstimatedLatencyNs.Should().BeInRange(1000.0, 10000.0);
    }

    [Fact]
    public void GetP2PCapability_ReportsNoAtomics()
    {
        // Act
        var capability = _provider.GetP2PCapability(0, 1);

        // Assert - CPU fallback doesn't support P2P atomics
        capability!.AtomicsSupported.Should().BeFalse();
        capability.NativeAtomicsSupported.Should().BeFalse();
    }

    #endregion
}
