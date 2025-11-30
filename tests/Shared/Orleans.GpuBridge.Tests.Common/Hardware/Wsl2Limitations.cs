// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Tests.Common.Hardware;

/// <summary>
/// Documents WSL2 GPU limitations for test categorization and skip logic.
/// </summary>
/// <remarks>
/// <para><strong>WSL2 GPU-PV Limitations:</strong></para>
/// <list type="bullet">
/// <item><description>No system-scope atomics - GPU kernel cannot see host memory writes in real-time</description></item>
/// <item><description>No persistent kernel mode - kernels cannot poll for host signals</description></item>
/// <item><description>No unified memory spill - datasets larger than VRAM fail</description></item>
/// <item><description>Higher latency (~5s) for host-GPU coordination vs native (~100-500ns)</description></item>
/// </list>
/// </remarks>
public static class Wsl2Limitations
{
    /// <summary>
    /// Features that work in WSL2.
    /// </summary>
    public static class Working
    {
        /// <summary>Basic CUDA kernel execution (launch, execute, return).</summary>
        public const string BasicKernels = "Basic CUDA kernels work";

        /// <summary>EventDriven mode for ring kernels (process available messages, terminate).</summary>
        public const string EventDrivenMode = "EventDriven ring kernel mode works";

        /// <summary>Host-side control before kernel launch.</summary>
        public const string HostSideControl = "Host-side control before launch works";

        /// <summary>Memory copies between host and device.</summary>
        public const string MemoryCopies = "Host-device memory copies work";
    }

    /// <summary>
    /// Features that fail or are unreliable in WSL2.
    /// </summary>
    public static class Failing
    {
        /// <summary>System-scope atomics are unreliable in WSL2.</summary>
        public const string SystemScopeAtomics = "System-scope atomics unreliable";

        /// <summary>Persistent kernel mode fails - kernel never sees host signal changes.</summary>
        public const string PersistentKernels = "Persistent kernels fail";

        /// <summary>Unified memory cannot spill from VRAM to system RAM.</summary>
        public const string UnifiedMemorySpill = "Unified memory spill unsupported";

        /// <summary>__threadfence_system() is unreliable.</summary>
        public const string ThreadfenceSystem = "__threadfence_system unreliable";
    }

    /// <summary>
    /// Performance comparison between WSL2 and native Linux.
    /// </summary>
    public static class Performance
    {
        /// <summary>WSL2 message latency is ~5 seconds due to event-driven relaunch.</summary>
        public const long Wsl2MessageLatencyMs = 5000;

        /// <summary>Native Linux message latency is 100-500 nanoseconds with persistent kernels.</summary>
        public const long NativeMessageLatencyNs = 500;

        /// <summary>Acceptable tolerance for timing tests in WSL2 (more relaxed).</summary>
        public const double Wsl2TimingToleranceMultiplier = 2.0;
    }
}
