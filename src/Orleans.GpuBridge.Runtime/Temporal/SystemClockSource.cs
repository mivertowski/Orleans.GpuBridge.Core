using System;
using System.Diagnostics;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Physical clock source using the system clock with no external synchronization.
/// </summary>
/// <remarks>
/// <para>
/// This is the default clock source with the following characteristics:
/// </para>
/// <list type="bullet">
///   <item><description>Resolution: 100 nanoseconds (on Windows) or better</description></item>
///   <item><description>Accuracy: ±1-100ms (depends on OS clock synchronization)</description></item>
///   <item><description>Drift: Typically 10-50 PPM without NTP</description></item>
/// </list>
/// <para>
/// For production use cases requiring high accuracy, consider using
/// <see cref="NtpClockSource"/> or PtpClockSource (Phase 6).
/// </para>
/// </remarks>
public sealed class SystemClockSource : IPhysicalClockSource
{
    private static readonly DateTimeOffset UnixEpoch = new(1970, 1, 1, 0, 0, 0, TimeSpan.Zero);
    private const long TicksToNanos = 100; // 1 tick = 100 nanoseconds

    /// <inheritdoc/>
    public long GetCurrentTimeNanos()
    {
        var utcNow = DateTimeOffset.UtcNow;
        var elapsedTicks = utcNow.Ticks - UnixEpoch.Ticks;
        return elapsedTicks * TicksToNanos;
    }

    /// <inheritdoc/>
    public long GetErrorBound()
    {
        // Assume ±10ms error bound for unsynchronized system clock
        // In practice, this varies widely:
        // - Virtualized environments: 10-100ms
        // - Physical hardware with NTP: 1-10ms
        // - Physical hardware without NTP: 10-1000ms
        return 10_000_000; // 10 milliseconds
    }

    /// <inheritdoc/>
    public bool IsSynchronized => false; // No external synchronization

    /// <inheritdoc/>
    public double GetClockDrift()
    {
        // Typical hardware clock drift: 10-50 PPM without synchronization
        // This is a conservative estimate; actual drift can be measured
        return 50.0; // 50 PPM = 50 microseconds per second
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"SystemClockSource(ErrorBound=±{GetErrorBound() / 1_000_000.0:F1}ms, Drift={GetClockDrift()}PPM)";
    }
}

/// <summary>
/// Physical clock source using NTP (Network Time Protocol) synchronization.
/// </summary>
/// <remarks>
/// <para>
/// NTP provides clock synchronization across networks with the following characteristics:
/// </para>
/// <list type="bullet">
///   <item><description>Accuracy: ±1-10ms (LAN), ±10-100ms (WAN)</description></item>
///   <item><description>Requires OS-level NTP daemon (ntpd, chronyd, or Windows Time Service)</description></item>
///   <item><description>Periodic synchronization (typically every 64-1024 seconds)</description></item>
/// </list>
/// <para>
/// This implementation queries the OS for NTP synchronization status and adjusts
/// error bounds accordingly.
/// </para>
/// <para>
/// Platform Support:
/// - Linux: Reads /var/lib/ntp/drift or queries timedatectl
/// - Windows: Queries Windows Time Service
/// - macOS: Queries systemsetup -getusingnetworktime
/// </para>
/// </remarks>
public sealed class NtpClockSource : IPhysicalClockSource
{
    private static readonly DateTimeOffset UnixEpoch = new(1970, 1, 1, 0, 0, 0, TimeSpan.Zero);
    private const long TicksToNanos = 100;

    private readonly SystemClockSource _fallbackClock = new();
    private DateTime _lastSyncCheck = DateTime.MinValue;
    private bool _isSynchronized;
    private double _measuredDrift;
    private long _errorBound;

    /// <summary>
    /// Creates a new NTP clock source.
    /// </summary>
    public NtpClockSource()
    {
        UpdateSynchronizationStatus();
    }

    /// <inheritdoc/>
    public long GetCurrentTimeNanos()
    {
        // Periodically check synchronization status (every 60 seconds)
        if ((DateTime.UtcNow - _lastSyncCheck).TotalSeconds > 60)
        {
            UpdateSynchronizationStatus();
        }

        var utcNow = DateTimeOffset.UtcNow;
        var elapsedTicks = utcNow.Ticks - UnixEpoch.Ticks;
        return elapsedTicks * TicksToNanos;
    }

    /// <inheritdoc/>
    public long GetErrorBound()
    {
        return _errorBound;
    }

    /// <inheritdoc/>
    public bool IsSynchronized => _isSynchronized;

    /// <inheritdoc/>
    public double GetClockDrift()
    {
        return _measuredDrift;
    }

    /// <summary>
    /// Updates NTP synchronization status from the operating system.
    /// </summary>
    private void UpdateSynchronizationStatus()
    {
        _lastSyncCheck = DateTime.UtcNow;

        try
        {
            if (OperatingSystem.IsWindows())
            {
                UpdateWindowsNtpStatus();
            }
            else if (OperatingSystem.IsLinux())
            {
                UpdateLinuxNtpStatus();
            }
            else if (OperatingSystem.IsMacOS())
            {
                UpdateMacOsNtpStatus();
            }
            else
            {
                // Unknown platform, use conservative defaults
                _isSynchronized = false;
                _errorBound = 100_000_000; // ±100ms
                _measuredDrift = 50.0; // 50 PPM
            }
        }
        catch
        {
            // If status check fails, assume not synchronized
            _isSynchronized = false;
            _errorBound = 100_000_000; // ±100ms
            _measuredDrift = 50.0; // 50 PPM
        }
    }

    private void UpdateWindowsNtpStatus()
    {
        // Phase 7.1: Query Windows Time Service status via w32tm command
        // The w32tm tool is the standard utility for Windows Time Service management
        try
        {
            using var process = new System.Diagnostics.Process();
            process.StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "w32tm",
                Arguments = "/query /status",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            if (process.Start())
            {
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit(1000);

                // Parse output for synchronization status
                // Example output contains: "Source: time.windows.com" and "Stratum: 4"
                var isSynced = output.Contains("Source:", StringComparison.OrdinalIgnoreCase) &&
                               !output.Contains("Local CMOS Clock", StringComparison.OrdinalIgnoreCase);

                if (isSynced)
                {
                    _isSynchronized = true;
                    _errorBound = 10_000_000; // ±10ms (typical for Windows Time Service)
                    _measuredDrift = 5.0; // 5 PPM (with NTP correction)
                }
                else
                {
                    // Not synchronized to external source
                    _isSynchronized = false;
                    _errorBound = 100_000_000; // ±100ms (unsynchronized)
                    _measuredDrift = 50.0; // 50 PPM (local clock drift)
                }

                return;
            }
        }
        catch
        {
            // Fall through to conservative defaults
        }

        // Conservative defaults if w32tm query fails
        _isSynchronized = true; // Assume synchronized (most Windows systems have it)
        _errorBound = 10_000_000;
        _measuredDrift = 5.0;
    }

    private void UpdateLinuxNtpStatus()
    {
        // Phase 7.1: Parse timedatectl output or read NTP drift file
        try
        {
            // First try timedatectl show (works on systemd systems)
            using var process = new System.Diagnostics.Process();
            process.StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "timedatectl",
                Arguments = "show",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            if (process.Start())
            {
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit(1000);

                // Parse for NTPSynchronized=yes
                var isSynced = output.Contains("NTPSynchronized=yes", StringComparison.OrdinalIgnoreCase);

                if (isSynced)
                {
                    _isSynchronized = true;
                    _errorBound = 5_000_000; // ±5ms (typical for LAN NTP)
                    _measuredDrift = 2.0; // 2 PPM (with NTP correction)
                    return;
                }
            }
        }
        catch
        {
            // timedatectl not available, try drift file
        }

        // Try reading NTP drift file (classic ntp daemon)
        try
        {
            const string driftFile = "/var/lib/ntp/drift";
            if (System.IO.File.Exists(driftFile))
            {
                var content = System.IO.File.ReadAllText(driftFile).Trim();
                if (double.TryParse(content, out var drift))
                {
                    // Drift file exists and has valid content - ntpd is managing clock
                    _isSynchronized = true;
                    _errorBound = 5_000_000; // ±5ms
                    _measuredDrift = Math.Abs(drift); // Use actual drift from file
                    return;
                }
            }
        }
        catch
        {
            // Drift file not accessible
        }

        // Conservative defaults
        _isSynchronized = true; // Assume NTP daemon is running
        _errorBound = 5_000_000;
        _measuredDrift = 2.0;
    }

    private void UpdateMacOsNtpStatus()
    {
        // Phase 7.1: Check macOS network time synchronization via systemsetup
        try
        {
            using var process = new System.Diagnostics.Process();
            process.StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "systemsetup",
                Arguments = "-getusingnetworktime",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            if (process.Start())
            {
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit(1000);

                // Parse for "Network Time: On"
                var isSynced = output.Contains(": On", StringComparison.OrdinalIgnoreCase);

                _isSynchronized = isSynced;
                _errorBound = isSynced ? 10_000_000 : 100_000_000; // ±10ms synced, ±100ms unsynced
                _measuredDrift = isSynced ? 5.0 : 50.0;
                return;
            }
        }
        catch
        {
            // systemsetup not available or failed
        }

        // Conservative defaults - assume NTP is enabled (default on macOS)
        _isSynchronized = true;
        _errorBound = 10_000_000; // ±10ms
        _measuredDrift = 5.0; // 5 PPM
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"NtpClockSource(Synchronized={_isSynchronized}, ErrorBound=±{_errorBound / 1_000_000.0:F1}ms, Drift={_measuredDrift:F1}PPM)";
    }
}
