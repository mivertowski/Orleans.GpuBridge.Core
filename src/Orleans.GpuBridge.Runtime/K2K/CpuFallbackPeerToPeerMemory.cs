// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.K2K;

namespace Orleans.GpuBridge.Runtime.K2K;

/// <summary>
/// CPU fallback implementation of P2P memory operations.
/// Uses CPU-staged memory transfers when true GPU P2P is not available.
/// </summary>
/// <remarks>
/// <para>
/// This implementation routes all P2P operations through CPU memory:
/// <list type="bullet">
/// <item><description>Copy from source GPU to CPU staging buffer</description></item>
/// <item><description>Copy from CPU staging buffer to destination GPU</description></item>
/// </list>
/// </para>
/// <para>
/// **Performance Characteristics:**
/// <list type="bullet">
/// <item><description>Bandwidth: 15-25 GB/s (limited by PCIe)</description></item>
/// <item><description>Latency: 2-10μs per transfer</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class CpuFallbackPeerToPeerMemory : IGpuPeerToPeerMemory
{
    private readonly ILogger<CpuFallbackPeerToPeerMemory> _logger;

    /// <summary>
    /// Initializes a new instance of the CPU fallback P2P memory implementation.
    /// </summary>
    /// <param name="logger">Logger for operations.</param>
    public CpuFallbackPeerToPeerMemory(ILogger<CpuFallbackPeerToPeerMemory> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _logger.LogInformation("Initialized CPU fallback P2P memory provider");
    }

    /// <inheritdoc />
    public bool CanAccessPeer(int sourceDeviceId, int targetDeviceId)
    {
        // CPU fallback always reports no direct P2P - it uses CPU staging
        return false;
    }

    /// <inheritdoc />
    public Task<bool> EnablePeerAccessAsync(
        int sourceDeviceId,
        int targetDeviceId,
        CancellationToken cancellationToken = default)
    {
        _logger.LogDebug(
            "EnablePeerAccess called for {Source} -> {Target} (CPU fallback, no-op)",
            sourceDeviceId, targetDeviceId);

        // CPU fallback doesn't need to enable anything
        return Task.FromResult(true);
    }

    /// <inheritdoc />
    public Task DisablePeerAccessAsync(
        int sourceDeviceId,
        int targetDeviceId,
        CancellationToken cancellationToken = default)
    {
        _logger.LogDebug(
            "DisablePeerAccess called for {Source} -> {Target} (CPU fallback, no-op)",
            sourceDeviceId, targetDeviceId);

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public Task CopyPeerToPeerAsync(
        IntPtr sourcePtr,
        int sourceDeviceId,
        IntPtr destinationPtr,
        int destinationDeviceId,
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        // CPU fallback: just do a direct memory copy (assumes unified/managed memory or CPU memory)
        CopyPeerToPeer(sourcePtr, sourceDeviceId, destinationPtr, destinationDeviceId, sizeBytes);
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void CopyPeerToPeer(
        IntPtr sourcePtr,
        int sourceDeviceId,
        IntPtr destinationPtr,
        int destinationDeviceId,
        long sizeBytes)
    {
        if (sourcePtr == IntPtr.Zero || destinationPtr == IntPtr.Zero)
        {
            throw new ArgumentException("Source and destination pointers cannot be null");
        }

        if (sizeBytes <= 0)
        {
            return;
        }

        // CPU fallback: direct memory copy (works for CPU memory or unified memory)
        // In production, this would go through GPU memory allocator for device-to-host-to-device
        Buffer.MemoryCopy(
            (void*)sourcePtr,
            (void*)destinationPtr,
            sizeBytes,
            sizeBytes);

        _logger.LogTrace(
            "CPU fallback P2P copy: {Source}[{SrcDev}] -> {Dest}[{DstDev}] ({Size} bytes)",
            sourcePtr.ToString("X16"), sourceDeviceId,
            destinationPtr.ToString("X16"), destinationDeviceId,
            sizeBytes);
    }

    /// <inheritdoc />
    public Task<IntPtr> MapPeerMemoryAsync(
        IntPtr localPtr,
        int localDeviceId,
        int remoteDeviceId,
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        // CPU fallback: no mapping needed, return the same pointer
        // In unified memory systems, the pointer is accessible from both devices
        _logger.LogDebug(
            "MapPeerMemory called for {Local}[{LocalDev}] -> [{RemoteDev}] (CPU fallback, returning same pointer)",
            localPtr.ToString("X16"), localDeviceId, remoteDeviceId);

        return Task.FromResult(localPtr);
    }

    /// <inheritdoc />
    public Task UnmapPeerMemoryAsync(
        IntPtr mappedPtr,
        int remoteDeviceId,
        CancellationToken cancellationToken = default)
    {
        // CPU fallback: no unmapping needed
        _logger.LogDebug(
            "UnmapPeerMemory called for {Ptr}[{Dev}] (CPU fallback, no-op)",
            mappedPtr.ToString("X16"), remoteDeviceId);

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public P2PCapabilityInfo? GetP2PCapability(int sourceDeviceId, int targetDeviceId)
    {
        // Return CPU-routed capability info
        return new P2PCapabilityInfo(
            SourceDeviceId: sourceDeviceId,
            TargetDeviceId: targetDeviceId,
            IsSupported: false,  // No true P2P
            IsEnabled: true,     // CPU routing is always enabled
            AccessType: P2PAccessType.None,
            EstimatedBandwidthGBps: 20.0,  // Typical PCIe bandwidth
            EstimatedLatencyNs: 5000.0,    // ~5μs for CPU staging
            AtomicsSupported: false,
            NativeAtomicsSupported: false);
    }
}
