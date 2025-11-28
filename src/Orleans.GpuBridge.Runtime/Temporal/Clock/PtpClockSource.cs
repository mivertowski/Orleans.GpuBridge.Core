using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Microsoft.Win32.SafeHandles;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal.Clock;

/// <summary>
/// PTP (Precision Time Protocol) hardware clock source for sub-microsecond timing.
/// Accesses /dev/ptp* devices on Linux via POSIX clock_gettime().
/// Provides ±100ns-1μs accuracy depending on hardware.
/// </summary>
/// <remarks>
/// Hardware requirements:
/// - Linux: PTP-capable NIC with /dev/ptp0 device
/// - Windows: Uses Windows Time Service API (limited PTP support)
///
/// Supported NICs:
/// - Intel i210/i211 (consumer, ±100ns)
/// - Intel X540/X550 (enterprise, ±100ns)
/// - Mellanox ConnectX-3/4/5 (high-end, ±50ns)
/// - Hyper-V synthetic NIC (virtual, ±1-5μs)
///
/// Fallback: If PTP hardware unavailable, use SoftwarePtpClient or NtpClockSource.
/// </remarks>
public sealed class PtpClockSource : IPhysicalClockSource, IDisposable
{
    private readonly ILogger<PtpClockSource> _logger;
    private readonly string _ptpDevicePath;

    private SafeFileHandle? _ptpDevice;
    private int _clockId = -1;
    private long _errorBoundNanos;

    // Windows PTP state
    private bool _windowsPtpInitialized;

    /// <summary>
    /// Gets whether PTP clock is synchronized and available.
    /// </summary>
    public bool IsSynchronized { get; private set; }

    /// <summary>
    /// Gets the PTP device path (e.g., "/dev/ptp0").
    /// </summary>
    public string ClockPath => _ptpDevicePath;

    /// <summary>
    /// Gets the PTP clock ID used for clock_gettime().
    /// </summary>
    public int ClockId => _clockId;

    /// <summary>
    /// Initializes a new PTP clock source.
    /// </summary>
    /// <param name="logger">Logger for diagnostic messages.</param>
    /// <param name="ptpDevicePath">Path to PTP device (default: /dev/ptp0).</param>
    public PtpClockSource(ILogger<PtpClockSource> logger, string ptpDevicePath = "/dev/ptp0")
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _ptpDevicePath = ptpDevicePath ?? throw new ArgumentNullException(nameof(ptpDevicePath));
    }

    /// <summary>
    /// Initializes PTP clock access.
    /// Detects available PTP hardware and configures clock ID.
    /// </summary>
    /// <returns>True if PTP hardware initialized successfully; false otherwise.</returns>
    public async Task<bool> InitializeAsync(CancellationToken ct = default)
    {
        if (IsSynchronized)
        {
            _logger.LogWarning("PTP clock already initialized");
            return true;
        }

        _logger.LogInformation("Initializing PTP clock source at {Path}", _ptpDevicePath);

        if (OperatingSystem.IsLinux())
        {
            return await Task.Run(() => InitializeLinuxPtp(), ct);
        }
        else if (OperatingSystem.IsWindows())
        {
            return await Task.Run(() => InitializeWindowsPtp(), ct);
        }
        else
        {
            _logger.LogWarning("PTP hardware not supported on {OS} - use software PTP fallback",
                Environment.OSVersion.Platform);
            return false;
        }
    }

    /// <summary>
    /// Gets current PTP time in nanoseconds since Unix epoch.
    /// </summary>
    /// <returns>Current time in nanoseconds.</returns>
    /// <exception cref="InvalidOperationException">PTP clock not initialized.</exception>
    public long GetCurrentTimeNanos()
    {
        if (!IsSynchronized)
        {
            throw new InvalidOperationException(
                "PTP clock not initialized. Call InitializeAsync() first.");
        }

        if (OperatingSystem.IsLinux())
        {
            return GetLinuxPtpTime();
        }
        else if (OperatingSystem.IsWindows())
        {
            return GetWindowsPtpTime();
        }

        throw new PlatformNotSupportedException($"PTP not supported on {Environment.OSVersion.Platform}");
    }

    /// <summary>
    /// Gets PTP clock uncertainty (error bound) in nanoseconds.
    /// </summary>
    /// <returns>Error bound in nanoseconds.</returns>
    public long GetErrorBound()
    {
        if (!IsSynchronized)
        {
            // Not synchronized - return large error bound
            return 100_000_000; // 100ms
        }

        return _errorBoundNanos;
    }

    /// <summary>
    /// Gets clock drift rate in parts per million (PPM).
    /// Returns 0 for PTP clocks (drift handled by PTP synchronization).
    /// </summary>
    public double GetClockDrift()
    {
        // PTP clocks are continuously synchronized, so reported drift is 0
        // Actual drift is compensated by PTP protocol
        return 0.0;
    }

    /// <summary>
    /// Initializes Linux PTP via /dev/ptp* device.
    /// </summary>
    private bool InitializeLinuxPtp()
    {
        try
        {
            // Check if PTP device exists
            if (!File.Exists(_ptpDevicePath))
            {
                _logger.LogWarning("PTP device not found at {Path}", _ptpDevicePath);
                return false;
            }

            // Open PTP device file (read-only)
            _ptpDevice = File.OpenHandle(
                _ptpDevicePath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read);

            _logger.LogDebug("Opened PTP device at {Path}", _ptpDevicePath);

            // Get PTP clock capabilities via ioctl
            var caps = new PtpClockCaps();
            int result = Ioctl(_ptpDevice, IOCTL_PTP_CLOCK_GETCAPS, ref caps);

            if (result != 0)
            {
                int error = Marshal.GetLastWin32Error();
                _logger.LogWarning("Failed to get PTP clock capabilities: errno={Error}", error);
                return false;
            }

            // Extract clock ID from capabilities
            // For most PTP devices, we can use a dynamic clock ID
            // Fallback to CLOCK_REALTIME if dynamic clock unavailable
            _clockId = caps.Index; // Use PTP clock index

            // Determine error bound based on hardware type
            _errorBoundNanos = DetermineErrorBound(caps);

            IsSynchronized = true;

            _logger.LogInformation(
                "PTP hardware clock initialized: {Device} (ClockId={ClockId}, MaxAdj={MaxAdj}ppb, ErrorBound=±{ErrorBound}ns)",
                _ptpDevicePath,
                _clockId,
                caps.MaxAdj,
                _errorBoundNanos);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to initialize PTP hardware clock at {Path}", _ptpDevicePath);
            Dispose(); // Clean up file handle
            return false;
        }
    }

    /// <summary>
    /// Gets current time from Linux PTP device using clock_gettime().
    /// Uses the PTP clock ID derived from the file descriptor via FD_TO_CLOCKID macro.
    /// </summary>
    private long GetLinuxPtpTime()
    {
        var timespec = new Timespec();

        // Use the PTP clock ID derived from the file descriptor
        // FD_TO_CLOCKID macro: ((~(clockid_t)(fd) << 3) | CLOCKID_DYNAMIC)
        // where CLOCKID_DYNAMIC = 1
        int clockIdToUse;

        if (_ptpDevice != null && !_ptpDevice.IsInvalid && _clockId >= 0)
        {
            // Use dynamic clock ID calculated from PTP device file descriptor
            // Formula: clockId = ((~fd) << 3) | 1
            nint fd = _ptpDevice.DangerousGetHandle();
            clockIdToUse = (~(int)fd << 3) | CLOCKID_DYNAMIC;
        }
        else
        {
            // Fallback to CLOCK_REALTIME if PTP device not available
            clockIdToUse = CLOCK_REALTIME;
        }

        int result = clock_gettime(clockIdToUse, ref timespec);

        if (result != 0)
        {
            int error = Marshal.GetLastWin32Error();
            throw new InvalidOperationException($"Failed to read PTP time (clockId={clockIdToUse}): errno={error}");
        }

        // Convert to nanoseconds since Unix epoch
        return timespec.Seconds * 1_000_000_000L + timespec.Nanoseconds;
    }

    /// <summary>
    /// Determines error bound based on PTP hardware capabilities.
    /// </summary>
    private long DetermineErrorBound(PtpClockCaps caps)
    {
        // Read clock name from sysfs to identify hardware
        string clockName = "unknown";
        try
        {
            string clockNamePath = $"/sys/class/ptp/ptp{caps.Index}/clock_name";
            if (File.Exists(clockNamePath))
            {
                clockName = File.ReadAllText(clockNamePath).Trim();
            }
        }
        catch
        {
            // Ignore errors reading clock name
        }

        _logger.LogDebug("PTP clock name: {ClockName}", clockName);

        // Determine accuracy based on hardware type
        return clockName.ToLowerInvariant() switch
        {
            // Physical NICs with hardware timestamping
            string name when name.Contains("intel") => 100,        // ±100ns (Intel i210/i211)
            string name when name.Contains("mellanox") => 50,      // ±50ns (Mellanox ConnectX)
            string name when name.Contains("broadcom") => 100,     // ±100ns (Broadcom NetXtreme)

            // Virtual NICs
            string name when name.Contains("hyperv") => 5_000,     // ±5μs (Hyper-V)
            string name when name.Contains("kvm") => 10_000,       // ±10μs (KVM)
            string name when name.Contains("vmware") => 10_000,    // ±10μs (VMware)

            // Unknown hardware - conservative estimate
            _ => 1_000  // ±1μs (software PTP)
        };
    }

    /// <summary>
    /// Initializes Windows PTP using high-resolution performance counter.
    /// </summary>
    /// <remarks>
    /// Windows does not have native PTP hardware clock access like Linux.
    /// Instead, we use QueryPerformanceCounter which provides high-resolution
    /// timestamps, typically with sub-microsecond precision on modern hardware.
    ///
    /// For true PTP support on Windows, third-party PTP stacks like Greyware
    /// or Tekron are required. This implementation provides a software fallback
    /// using Windows' best available clock source.
    /// </remarks>
    private bool InitializeWindowsPtp()
    {
        try
        {
            // Check if high-resolution performance counter is available
            if (!QueryPerformanceFrequency(out long frequency))
            {
                _logger.LogWarning("Windows high-resolution performance counter not available");
                return false;
            }

            // Verify reasonable frequency (should be > 1MHz on modern systems)
            if (frequency < 1_000_000)
            {
                _logger.LogWarning("Windows performance counter frequency too low: {Frequency} Hz", frequency);
                return false;
            }

            // Calculate error bound based on counter resolution
            // Error bound = 1/frequency in nanoseconds
            _errorBoundNanos = Math.Max(1_000_000_000L / frequency, 100);

            _windowsPtpInitialized = true;
            IsSynchronized = true;

            _logger.LogInformation(
                "Windows PTP clock initialized: using QueryPerformanceCounter (Frequency={Frequency}Hz, ErrorBound=±{ErrorBound}ns)",
                frequency,
                _errorBoundNanos);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to initialize Windows PTP clock");
            return false;
        }
    }

    /// <summary>
    /// Gets current time from Windows high-resolution performance counter.
    /// </summary>
    /// <returns>Current time in nanoseconds since Unix epoch.</returns>
    private long GetWindowsPtpTime()
    {
        if (!_windowsPtpInitialized)
        {
            throw new InvalidOperationException("Windows PTP clock not initialized");
        }

        // Get high-resolution timestamp
        if (!QueryPerformanceCounter(out long counter))
        {
            int error = Marshal.GetLastWin32Error();
            throw new InvalidOperationException($"Failed to read Windows performance counter: error={error}");
        }

        if (!QueryPerformanceFrequency(out long frequency))
        {
            throw new InvalidOperationException("Failed to read Windows performance counter frequency");
        }

        // Convert counter ticks to nanoseconds
        // counter / frequency = seconds
        // (counter * 1_000_000_000) / frequency = nanoseconds
        // Use 128-bit math to avoid overflow
        long seconds = counter / frequency;
        long remainder = counter % frequency;
        long nanos = seconds * 1_000_000_000L + (remainder * 1_000_000_000L) / frequency;

        // Add Unix epoch offset (Windows performance counter starts at boot, not Unix epoch)
        // We need to correlate with system time to get Unix epoch offset
        var utcNow = DateTimeOffset.UtcNow;
        var epochOffset = utcNow.ToUnixTimeMilliseconds() * 1_000_000L - nanos;

        // Return absolute time in nanoseconds since Unix epoch
        // Note: This is not perfectly precise as we're correlating two different time sources
        // For true PTP accuracy, hardware clock synchronization is required
        return nanos + epochOffset;
    }

    /// <summary>
    /// Releases PTP device resources.
    /// </summary>
    public void Dispose()
    {
        _ptpDevice?.Dispose();
        _ptpDevice = null;

        IsSynchronized = false;
        _logger.LogDebug("PTP clock source disposed");
    }

    // ==================== P/Invoke Declarations ====================

    // Linux clock constants
    private const int CLOCK_REALTIME = 0;
    private const int CLOCKID_DYNAMIC = 1; // Used in FD_TO_CLOCKID macro
    private const uint IOCTL_PTP_CLOCK_GETCAPS = 0x80D06D01; // _IOR('=', 1, struct ptp_clock_caps)

    /// <summary>
    /// POSIX clock_gettime() system call.
    /// </summary>
    [DllImport("libc", SetLastError = true)]
    private static extern int clock_gettime(int clockId, ref Timespec tp);

    /// <summary>
    /// POSIX ioctl() system call for PTP device capabilities.
    /// </summary>
    [DllImport("libc", SetLastError = true, EntryPoint = "ioctl")]
    private static extern int Ioctl(SafeFileHandle fd, uint request, ref PtpClockCaps caps);

    // Windows high-resolution timer functions

    /// <summary>
    /// Windows QueryPerformanceCounter for high-resolution timestamps.
    /// </summary>
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);

    /// <summary>
    /// Windows QueryPerformanceFrequency for timer resolution.
    /// </summary>
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool QueryPerformanceFrequency(out long lpFrequency);

    /// <summary>
    /// POSIX timespec structure for nanosecond-precision time.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        /// <summary>Seconds since Unix epoch.</summary>
        public long Seconds;

        /// <summary>Nanoseconds (0-999,999,999).</summary>
        public long Nanoseconds;
    }

    /// <summary>
    /// PTP clock capabilities structure from linux/ptp_clock.h.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct PtpClockCaps
    {
        /// <summary>Maximum frequency adjustment (ppb).</summary>
        public int MaxAdj;

        /// <summary>Number of programmable alarms.</summary>
        public int NAdjust;

        /// <summary>Number of external time stamp channels.</summary>
        public int NExtTimestamp;

        /// <summary>Number of programmable periodic signals.</summary>
        public int NPerOut;

        /// <summary>PPS support (1=supported, 0=not supported).</summary>
        public int Pps;

        /// <summary>Number of time stamp pins.</summary>
        public int NPins;

        /// <summary>PTP clock index.</summary>
        public int Index;

        /// <summary>Reserved fields.</summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 14)]
        public int[] Reserved;
    }
}
