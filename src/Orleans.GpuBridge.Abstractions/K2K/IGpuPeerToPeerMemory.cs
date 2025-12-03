// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.K2K;

/// <summary>
/// Interface for GPU peer-to-peer (P2P) memory operations.
/// Enables direct GPU-to-GPU memory access without CPU involvement.
/// </summary>
/// <remarks>
/// <para>
/// P2P memory access allows GPUs to directly read/write each other's memory,
/// enabling sub-microsecond latency communication between GPU-resident actors.
/// </para>
/// <para>
/// **Performance Characteristics:**
/// <list type="bullet">
/// <item><description>NVLink: 600+ GB/s bandwidth, 100-200ns latency</description></item>
/// <item><description>PCIe P2P: 32-64 GB/s bandwidth, 500ns-1μs latency</description></item>
/// <item><description>CPU-routed (fallback): 15-25 GB/s bandwidth, 2-10μs latency</description></item>
/// </list>
/// </para>
/// <para>
/// **Implementation Notes:**
/// - CUDA: Uses cuMemcpyPeer, cuCtxEnablePeerAccess
/// - ROCm: Uses hipMemcpyPeer, hipDeviceEnablePeerAccess
/// - DotCompute: Backend-specific implementation
/// </para>
/// </remarks>
public interface IGpuPeerToPeerMemory
{
    /// <summary>
    /// Checks if peer-to-peer access is supported between two GPU devices.
    /// </summary>
    /// <param name="sourceDeviceId">The source GPU device ID.</param>
    /// <param name="targetDeviceId">The target GPU device ID.</param>
    /// <returns>True if P2P access is supported between the devices.</returns>
    bool CanAccessPeer(int sourceDeviceId, int targetDeviceId);

    /// <summary>
    /// Enables peer-to-peer access from source device to target device.
    /// </summary>
    /// <param name="sourceDeviceId">The source GPU device ID.</param>
    /// <param name="targetDeviceId">The target GPU device ID.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if P2P access was enabled successfully.</returns>
    /// <exception cref="InvalidOperationException">If P2P is not supported between devices.</exception>
    Task<bool> EnablePeerAccessAsync(
        int sourceDeviceId,
        int targetDeviceId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Disables peer-to-peer access from source device to target device.
    /// </summary>
    /// <param name="sourceDeviceId">The source GPU device ID.</param>
    /// <param name="targetDeviceId">The target GPU device ID.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task DisablePeerAccessAsync(
        int sourceDeviceId,
        int targetDeviceId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Copies memory directly between two GPU devices using P2P transfer.
    /// </summary>
    /// <param name="sourcePtr">Source memory pointer (on source device).</param>
    /// <param name="sourceDeviceId">Source device ID.</param>
    /// <param name="destinationPtr">Destination memory pointer (on target device).</param>
    /// <param name="destinationDeviceId">Destination device ID.</param>
    /// <param name="sizeBytes">Number of bytes to copy.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Task that completes when the copy is done.</returns>
    Task CopyPeerToPeerAsync(
        IntPtr sourcePtr,
        int sourceDeviceId,
        IntPtr destinationPtr,
        int destinationDeviceId,
        long sizeBytes,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Copies memory directly between two GPU devices synchronously.
    /// </summary>
    /// <remarks>
    /// Use this for small, latency-critical transfers where async overhead is undesirable.
    /// </remarks>
    void CopyPeerToPeer(
        IntPtr sourcePtr,
        int sourceDeviceId,
        IntPtr destinationPtr,
        int destinationDeviceId,
        long sizeBytes);

    /// <summary>
    /// Maps a device memory pointer to be accessible from another device.
    /// </summary>
    /// <param name="localPtr">Local memory pointer.</param>
    /// <param name="localDeviceId">Local device ID where memory is allocated.</param>
    /// <param name="remoteDeviceId">Remote device ID that needs access.</param>
    /// <param name="sizeBytes">Size of the memory region to map.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Pointer accessible from the remote device.</returns>
    Task<IntPtr> MapPeerMemoryAsync(
        IntPtr localPtr,
        int localDeviceId,
        int remoteDeviceId,
        long sizeBytes,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Unmaps previously mapped peer memory.
    /// </summary>
    /// <param name="mappedPtr">The mapped pointer returned by MapPeerMemoryAsync.</param>
    /// <param name="remoteDeviceId">The remote device ID.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task UnmapPeerMemoryAsync(
        IntPtr mappedPtr,
        int remoteDeviceId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets detailed P2P capability information between devices.
    /// </summary>
    /// <param name="sourceDeviceId">Source device ID.</param>
    /// <param name="targetDeviceId">Target device ID.</param>
    /// <returns>P2P capability details, or null if not available.</returns>
    P2PCapabilityInfo? GetP2PCapability(int sourceDeviceId, int targetDeviceId);
}

/// <summary>
/// Detailed information about P2P capabilities between two devices.
/// </summary>
/// <param name="SourceDeviceId">Source GPU device ID.</param>
/// <param name="TargetDeviceId">Target GPU device ID.</param>
/// <param name="IsSupported">Whether P2P is supported.</param>
/// <param name="IsEnabled">Whether P2P is currently enabled.</param>
/// <param name="AccessType">The type of P2P access available.</param>
/// <param name="EstimatedBandwidthGBps">Estimated bandwidth in GB/s.</param>
/// <param name="EstimatedLatencyNs">Estimated latency in nanoseconds.</param>
/// <param name="AtomicsSupported">Whether atomic operations are supported over P2P.</param>
/// <param name="NativeAtomicsSupported">Whether native (non-emulated) atomics are supported.</param>
public sealed record P2PCapabilityInfo(
    int SourceDeviceId,
    int TargetDeviceId,
    bool IsSupported,
    bool IsEnabled,
    P2PAccessType AccessType,
    double EstimatedBandwidthGBps,
    double EstimatedLatencyNs,
    bool AtomicsSupported,
    bool NativeAtomicsSupported);

/// <summary>
/// Type of peer-to-peer access between GPUs.
/// </summary>
public enum P2PAccessType
{
    /// <summary>
    /// No P2P access available - must use CPU staging.
    /// </summary>
    None = 0,

    /// <summary>
    /// P2P over PCIe bus.
    /// </summary>
    PciExpress = 1,

    /// <summary>
    /// P2P over NVLink (NVIDIA high-speed interconnect).
    /// </summary>
    NvLink = 2,

    /// <summary>
    /// P2P over AMD Infinity Fabric.
    /// </summary>
    InfinityFabric = 3,

    /// <summary>
    /// P2P over Intel Xe Link.
    /// </summary>
    XeLink = 4,

    /// <summary>
    /// Unified memory with coherent access.
    /// </summary>
    UnifiedMemory = 5
}

/// <summary>
/// GPU direct messaging mode for K2K communication.
/// </summary>
public enum GpuDirectMessagingMode
{
    /// <summary>
    /// All messages route through CPU (default, always works).
    /// Latency: 2-10μs per message.
    /// </summary>
    CpuRouted = 0,

    /// <summary>
    /// Prefer P2P when available, fall back to CPU routing.
    /// Automatically detects and uses the best available path.
    /// </summary>
    PreferP2P = 1,

    /// <summary>
    /// Use P2P over PCIe when devices support it.
    /// Latency: 500ns-1μs per message.
    /// </summary>
    PciExpressP2P = 2,

    /// <summary>
    /// Use NVLink for NVIDIA GPUs with NVLink connectivity.
    /// Latency: 100-200ns per message.
    /// </summary>
    NvLink = 3,

    /// <summary>
    /// Use AMD Infinity Fabric for AMD GPUs.
    /// Latency: 100-300ns per message.
    /// </summary>
    InfinityFabric = 4,

    /// <summary>
    /// Use GPUDirect RDMA for cross-node communication.
    /// Requires InfiniBand or RoCE network adapters.
    /// </summary>
    GpuDirectRdma = 5
}
