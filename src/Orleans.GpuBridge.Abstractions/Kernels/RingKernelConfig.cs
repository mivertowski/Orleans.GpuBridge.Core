using Orleans.GpuBridge.Abstractions.K2K;

namespace Orleans.GpuBridge.Abstractions.Kernels;

/// <summary>
/// Configuration for GPU-resident ring kernel actors.
/// Ring kernels are persistent GPU threads that process messages at sub-microsecond latencies.
/// </summary>
/// <remarks>
/// **GPU-Native Actor Model:**
/// - Actors reside permanently in GPU memory
/// - Ring kernel runs infinite dispatch loop (launched once, runs forever)
/// - Message queue and state maintained entirely on GPU
/// - 100-500ns message latency (vs 10-100μs for CPU actors)
///
/// **Use Cases:**
/// - High-frequency messaging between actors
/// - Real-time temporal graph pattern detection
/// - Digital twins with physics simulation
/// - Knowledge organisms with emergent intelligence
/// </remarks>
public sealed record RingKernelConfig
{
    /// <summary>
    /// Message queue depth (power of 2 recommended for lock-free operations)
    /// </summary>
    public int QueueDepth { get; init; } = 1024;

    /// <summary>
    /// Enable GPU-resident Hybrid Logical Clock for temporal ordering
    /// </summary>
    public bool EnableHLC { get; init; } = true;

    /// <summary>
    /// Keep actor state permanently in GPU memory (vs CPU-GPU transfers)
    /// </summary>
    public bool GPUResident { get; init; } = true;

    /// <summary>
    /// Enable vector clock for distributed causality tracking
    /// </summary>
    public bool EnableVectorClock { get; init; } = false;

    /// <summary>
    /// Number of vector clock entries (for distributed causality)
    /// </summary>
    public int VectorClockSize { get; init; } = 16;

    /// <summary>
    /// Enable temporal pattern detection on message processing
    /// </summary>
    public bool EnableTemporalPatterns { get; init; } = false;

    /// <summary>
    /// Maximum message processing time before timeout (microseconds)
    /// </summary>
    public long MessageTimeoutMicroseconds { get; init; } = 1000; // 1ms

    /// <summary>
    /// Ring kernel dispatch loop polling interval (nanoseconds)
    /// Lower = lower latency, higher = better GPU efficiency
    /// </summary>
    public long PollingIntervalNanoseconds { get; init; } = 100; // 100ns polling

    /// <summary>
    /// Enable GPU-to-GPU direct messaging (bypasses CPU)
    /// Requires GPUDirect or similar technology
    /// </summary>
    public bool EnableGpuDirectMessaging { get; init; } = false;

    /// <summary>
    /// GPU direct messaging mode for K2K communication.
    /// Controls how messages are routed between GPU-resident actors.
    /// </summary>
    public GpuDirectMessagingMode MessagingMode { get; init; } = GpuDirectMessagingMode.CpuRouted;

    /// <summary>
    /// Enable automatic P2P path discovery and setup.
    /// When enabled, automatically detects and enables P2P access between GPUs.
    /// </summary>
    public bool AutoEnableP2P { get; init; } = true;

    /// <summary>
    /// Minimum bandwidth threshold (GB/s) to prefer P2P over CPU routing.
    /// P2P paths with lower bandwidth will fall back to CPU routing.
    /// </summary>
    public double P2PMinBandwidthGBps { get; init; } = 10.0;

    /// <summary>
    /// Maximum latency threshold (nanoseconds) to prefer P2P over CPU routing.
    /// P2P paths with higher latency will fall back to CPU routing.
    /// </summary>
    public double P2PMaxLatencyNs { get; init; } = 2000.0; // 2μs

    /// <summary>
    /// Enable P2P atomics for lock-free queue operations.
    /// Requires hardware support (NVLink, certain PCIe configurations).
    /// </summary>
    public bool EnableP2PAtomics { get; init; } = false;

    /// <summary>
    /// Enable persistent storage integration (GPUDirect Storage)
    /// </summary>
    public bool EnablePersistentStorage { get; init; } = false;

    /// <summary>
    /// Maximum state size in bytes (must fit in GPU memory)
    /// </summary>
    public long MaxStateSizeBytes { get; init; } = 1024 * 1024; // 1MB default

    /// <summary>
    /// Default configuration for standard ring kernel actors
    /// </summary>
    public static RingKernelConfig Default => new();

    /// <summary>
    /// High-performance configuration for low-latency messaging
    /// </summary>
    public static RingKernelConfig HighPerformance => new()
    {
        QueueDepth = 4096,
        EnableHLC = true,
        GPUResident = true,
        PollingIntervalNanoseconds = 50, // 50ns polling for ultra-low latency
        EnableGpuDirectMessaging = true,
        MessagingMode = GpuDirectMessagingMode.PreferP2P,
        AutoEnableP2P = true,
        EnableP2PAtomics = true,
        MessageTimeoutMicroseconds = 500 // 500μs timeout
    };

    /// <summary>
    /// Configuration for temporal graph pattern detection
    /// </summary>
    public static RingKernelConfig TemporalGraph => new()
    {
        QueueDepth = 2048,
        EnableHLC = true,
        EnableVectorClock = true,
        VectorClockSize = 32,
        EnableTemporalPatterns = true,
        GPUResident = true,
        PollingIntervalNanoseconds = 100
    };

    /// <summary>
    /// Configuration for knowledge organism actors
    /// </summary>
    public static RingKernelConfig KnowledgeOrganism => new()
    {
        QueueDepth = 8192,
        EnableHLC = true,
        EnableVectorClock = true,
        VectorClockSize = 64, // Support larger distributed systems
        EnableTemporalPatterns = true,
        GPUResident = true,
        MaxStateSizeBytes = 10 * 1024 * 1024, // 10MB for complex state
        PollingIntervalNanoseconds = 200 // Balance latency and efficiency
    };
}
