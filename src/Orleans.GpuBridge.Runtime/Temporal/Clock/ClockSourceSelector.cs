using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal.Clock;

/// <summary>
/// Automatically selects best available clock source with fallback chain.
/// Preference order: GPS → PTP Hardware → PTP Software → NTP → System Clock.
/// </summary>
/// <remarks>
/// Clock source accuracy comparison:
/// - GPS: ±50ns (requires GPS receiver)
/// - PTP Hardware: ±100ns-1μs (requires PTP-capable NIC)
/// - PTP Software: ±1-5μs (software implementation)
/// - NTP Client: ±10ms (Internet sync)
/// - System Clock: ±100ms (fallback, always available)
///
/// The selector automatically falls back to the next best source if initialization fails.
/// </remarks>
public sealed class ClockSourceSelector
{
    private readonly ILogger<ClockSourceSelector> _logger;
    private readonly List<IPhysicalClockSource> _availableSources = new();
    private IPhysicalClockSource? _activeSource;

    /// <summary>
    /// Gets the currently active clock source.
    /// </summary>
    /// <exception cref="InvalidOperationException">No clock source available.</exception>
    public IPhysicalClockSource ActiveSource => _activeSource ??
        throw new InvalidOperationException("No clock source available. Call InitializeAsync() first.");

    /// <summary>
    /// Gets all available clock sources in preference order.
    /// </summary>
    public IReadOnlyList<IPhysicalClockSource> AvailableSources => _availableSources.AsReadOnly();

    /// <summary>
    /// Gets whether initialization has been performed.
    /// </summary>
    public bool IsInitialized => _activeSource != null;

    /// <summary>
    /// Initializes a new clock source selector.
    /// </summary>
    /// <param name="logger">Logger for diagnostic messages.</param>
    public ClockSourceSelector(ILogger<ClockSourceSelector> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Initializes and selects best available clock source.
    /// Tries sources in preference order until one succeeds.
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        if (IsInitialized)
        {
            _logger.LogWarning("Clock source selector already initialized");
            return;
        }

        _logger.LogInformation("Detecting available clock sources...");

        // Try clock sources in order of preference (best accuracy first)
        await TryInitializeGpsClock(ct);
        await TryInitializePtpHardware(ct);
        // Software PTP requires server address - skip for now
        // await TryInitializePtpSoftware(ct);
        await TryInitializeNtpClient(ct);
        await TryInitializeSystemClock(ct);

        if (_activeSource == null)
        {
            throw new InvalidOperationException(
                "No clock source available - at least system clock should work");
        }

        _logger.LogInformation(
            "Clock source selected: {Source} (Accuracy: ±{ErrorBound}ns, Available sources: {Count})",
            _activeSource.GetType().Name,
            _activeSource.GetErrorBound(),
            _availableSources.Count);
    }

    /// <summary>
    /// Switches to a different clock source at runtime.
    /// </summary>
    /// <param name="newSource">New clock source to activate.</param>
    /// <exception cref="ArgumentException">Source not in available sources list.</exception>
    public void SwitchClockSource(IPhysicalClockSource newSource)
    {
        if (!_availableSources.Contains(newSource))
        {
            throw new ArgumentException("Clock source not in available sources list", nameof(newSource));
        }

        var oldSource = _activeSource;
        _activeSource = newSource;

        _logger.LogInformation(
            "Switched clock source: {OldSource} → {NewSource} (Accuracy: ±{OldError}ns → ±{NewError}ns)",
            oldSource?.GetType().Name ?? "None",
            newSource.GetType().Name,
            oldSource?.GetErrorBound() ?? 0,
            newSource.GetErrorBound());
    }

    /// <summary>
    /// Gets the best available clock source (without activating it).
    /// </summary>
    /// <returns>Clock source with lowest error bound.</returns>
    public IPhysicalClockSource? GetBestAvailableSource()
    {
        IPhysicalClockSource? best = null;
        long bestErrorBound = long.MaxValue;

        foreach (var source in _availableSources)
        {
            if (source.IsSynchronized)
            {
                long errorBound = source.GetErrorBound();
                if (errorBound < bestErrorBound)
                {
                    best = source;
                    bestErrorBound = errorBound;
                }
            }
        }

        return best;
    }

    private Task TryInitializeGpsClock(CancellationToken ct)
    {
        try
        {
            // GPS clock requires hardware receiver - typically not available
            // Placeholder for future GPS support
            _logger.LogDebug("GPS clock not implemented - skipping");
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "GPS clock not available");
            return Task.CompletedTask;
        }
    }

    private async Task TryInitializePtpHardware(CancellationToken ct)
    {
        if (_activeSource != null) return; // Already have better source

        try
        {
            var ptpClock = new PtpClockSource(
                Microsoft.Extensions.Logging.Abstractions.NullLogger<PtpClockSource>.Instance);

            if (await ptpClock.InitializeAsync(ct))
            {
                _availableSources.Add(ptpClock);
                _activeSource = ptpClock;

                _logger.LogInformation(
                    "PTP hardware clock available at {Path} (±{ErrorBound}ns accuracy)",
                    ptpClock.ClockPath,
                    ptpClock.GetErrorBound());
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "PTP hardware clock not available");
        }
    }

    private Task TryInitializeNtpClient(CancellationToken ct)
    {
        if (_activeSource != null) return Task.CompletedTask; // Already have better source

        try
        {
            // Use the working NtpClockSource from Orleans.GpuBridge.Runtime.Temporal namespace
            var ntpClock = new Orleans.GpuBridge.Runtime.Temporal.NtpClockSource();

            // NtpClockSource automatically checks OS NTP status on construction
            if (ntpClock.IsSynchronized)
            {
                _availableSources.Add(ntpClock);
                _activeSource = ntpClock;

                _logger.LogInformation(
                    "NTP clock source available (±{ErrorBound}ms accuracy, Drift: {Drift} PPM)",
                    ntpClock.GetErrorBound() / 1_000_000.0,
                    ntpClock.GetClockDrift());
            }
            else
            {
                // Add as fallback but don't activate
                _availableSources.Add(ntpClock);
                _logger.LogDebug("NTP clock available but not synchronized - added as fallback");
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "NTP client not available");
        }

        return Task.CompletedTask;
    }

    private Task TryInitializeSystemClock(CancellationToken ct)
    {
        // System clock always available as last resort
        var systemClock = new SystemClockSource(
            Microsoft.Extensions.Logging.Abstractions.NullLogger<SystemClockSource>.Instance);

        _availableSources.Add(systemClock);
        _activeSource ??= systemClock;

        _logger.LogInformation(
            "System clock available (±{ErrorBound}ms accuracy)",
            systemClock.GetErrorBound() / 1_000_000);

        return Task.CompletedTask;
    }
}

/// <summary>
/// System clock fallback (always available but least accurate).
/// Uses DateTimeOffset.UtcNow which may drift over time without synchronization.
/// </summary>
internal sealed class SystemClockSource : IPhysicalClockSource
{
    private readonly ILogger<SystemClockSource> _logger;

    /// <summary>
    /// System clock is always synchronized (though may drift).
    /// </summary>
    public bool IsSynchronized => true;

    /// <summary>
    /// Initializes a new system clock source.
    /// </summary>
    public SystemClockSource(ILogger<SystemClockSource> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Gets current system time in nanoseconds since Unix epoch.
    /// </summary>
    public long GetCurrentTimeNanos()
    {
        // DateTimeOffset.UtcNow has ~15ms resolution on Windows, ~1ms on Linux
        return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000;
    }

    /// <summary>
    /// Gets error bound for system clock (conservative: ±100ms).
    /// </summary>
    public long GetErrorBound()
    {
        // System clock can drift significantly without NTP sync
        // Conservative estimate: ±100ms
        return 100_000_000; // 100ms in nanoseconds
    }

    /// <summary>
    /// Gets clock drift rate in parts per million (PPM).
    /// System clock drift varies by hardware (typically 10-100 PPM).
    /// </summary>
    public double GetClockDrift()
    {
        // Typical crystal oscillator drift: 50 PPM
        // Without NTP sync, this accumulates over time
        return 50.0; // 50 PPM
    }
}

// NtpClockSource implementation is in Orleans.GpuBridge.Runtime.Temporal.NtpClockSource
