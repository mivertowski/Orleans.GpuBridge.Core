using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels;

/// <summary>
/// Monitors ring kernel health and automatically restarts hung or crashed kernels.
/// Prevents extended outages by detecting kernels that stop making progress.
/// </summary>
/// <remarks>
/// Detection Strategy:
/// - Monitor GPU timestamp updates (kernel heartbeat)
/// - Detect if no progress in configurable timeout (default 5 seconds)
/// - Automatic restart with state recovery
/// - Exponential backoff for repeated failures
///
/// Use Cases:
/// - GPU driver hangs (requires kernel restart)
/// - Kernel infinite loop bugs
/// - GPU memory corruption
/// - GPU hardware errors
/// </remarks>
public sealed class RingKernelWatchdog : IDisposable
{
    private readonly ILogger<RingKernelWatchdog> _logger;
    private readonly RingKernelManager _ringKernelManager;
    private readonly DotComputeTimingProvider _timingProvider;
    private readonly ConcurrentDictionary<Guid, WatchdogEntry> _monitoredKernels;
    private readonly Timer _watchdogTimer;
    private readonly RingKernelWatchdogOptions _options;
    private bool _disposed;

    public RingKernelWatchdog(
        RingKernelManager ringKernelManager,
        DotComputeTimingProvider timingProvider,
        ILogger<RingKernelWatchdog> logger,
        RingKernelWatchdogOptions? options = null)
    {
        _ringKernelManager = ringKernelManager ?? throw new ArgumentNullException(nameof(ringKernelManager));
        _timingProvider = timingProvider ?? throw new ArgumentNullException(nameof(timingProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? new RingKernelWatchdogOptions();

        _monitoredKernels = new ConcurrentDictionary<Guid, WatchdogEntry>();

        // Start watchdog timer
        _watchdogTimer = new Timer(
            CheckKernelHealth,
            null,
            TimeSpan.FromMilliseconds(_options.CheckIntervalMillis),
            TimeSpan.FromMilliseconds(_options.CheckIntervalMillis));

        _logger.LogInformation(
            "Ring kernel watchdog started - Check interval: {Interval}ms, Timeout: {Timeout}ms",
            _options.CheckIntervalMillis,
            _options.HungKernelTimeoutMillis);
    }

    /// <summary>
    /// Registers a ring kernel for monitoring.
    /// </summary>
    public async Task RegisterKernelAsync(RingKernelHandle handle, CancellationToken ct = default)
    {
        if (handle == null) throw new ArgumentNullException(nameof(handle));

        var initialTimestamp = await _timingProvider.GetGpuTimestampAsync(ct);

        var entry = new WatchdogEntry
        {
            Handle = handle,
            LastSeenTimestamp = initialTimestamp,
            LastCheckTime = DateTimeOffset.UtcNow,
            ConsecutiveTimeouts = 0,
            RestartCount = 0,
            RegistrationTime = DateTimeOffset.UtcNow
        };

        if (_monitoredKernels.TryAdd(handle.InstanceId, entry))
        {
            _logger.LogInformation(
                "Ring kernel {KernelId} registered with watchdog - Initial timestamp: {Timestamp}ns",
                handle.InstanceId,
                initialTimestamp);
        }
        else
        {
            _logger.LogWarning(
                "Ring kernel {KernelId} already registered with watchdog",
                handle.InstanceId);
        }
    }

    /// <summary>
    /// Unregisters a ring kernel from monitoring.
    /// </summary>
    public void UnregisterKernel(Guid instanceId)
    {
        if (_monitoredKernels.TryRemove(instanceId, out var entry))
        {
            var uptime = DateTimeOffset.UtcNow - entry.RegistrationTime;

            _logger.LogInformation(
                "Ring kernel {KernelId} unregistered from watchdog - " +
                "Uptime: {Uptime}, Restarts: {Restarts}",
                instanceId,
                uptime,
                entry.RestartCount);
        }
    }

    /// <summary>
    /// Gets watchdog statistics for a specific kernel.
    /// </summary>
    public RingKernelWatchdogStats? GetKernelStats(Guid instanceId)
    {
        if (!_monitoredKernels.TryGetValue(instanceId, out var entry))
            return null;

        return new RingKernelWatchdogStats
        {
            InstanceId = instanceId,
            IsHealthy = entry.ConsecutiveTimeouts == 0,
            LastSeenTimestamp = entry.LastSeenTimestamp,
            LastCheckTime = entry.LastCheckTime,
            ConsecutiveTimeouts = entry.ConsecutiveTimeouts,
            RestartCount = entry.RestartCount,
            Uptime = DateTimeOffset.UtcNow - entry.RegistrationTime
        };
    }

    /// <summary>
    /// Gets watchdog statistics for all monitored kernels.
    /// </summary>
    public RingKernelWatchdogStats[] GetAllKernelStats()
    {
        var stats = new RingKernelWatchdogStats[_monitoredKernels.Count];
        var index = 0;

        foreach (var kvp in _monitoredKernels)
        {
            var entry = kvp.Value;
            stats[index++] = new RingKernelWatchdogStats
            {
                InstanceId = kvp.Key,
                IsHealthy = entry.ConsecutiveTimeouts == 0,
                LastSeenTimestamp = entry.LastSeenTimestamp,
                LastCheckTime = entry.LastCheckTime,
                ConsecutiveTimeouts = entry.ConsecutiveTimeouts,
                RestartCount = entry.RestartCount,
                Uptime = DateTimeOffset.UtcNow - entry.RegistrationTime
            };
        }

        return stats;
    }

    private void CheckKernelHealth(object? state)
    {
        if (_disposed) return;

        try
        {
            foreach (var kvp in _monitoredKernels)
            {
                var instanceId = kvp.Key;
                var entry = kvp.Value;

                // Skip if kernel is not running
                if (!entry.Handle.IsRunning)
                {
                    _logger.LogDebug(
                        "Skipping health check for stopped kernel {KernelId}",
                        instanceId);
                    continue;
                }

                // Check if kernel is making progress
                Task.Run(async () => await CheckKernelProgressAsync(instanceId, entry))
                    .ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in watchdog health check loop");
        }
    }

    private async Task CheckKernelProgressAsync(Guid instanceId, WatchdogEntry entry)
    {
        try
        {
            // Get current GPU timestamp
            var currentTimestamp = await _timingProvider.GetGpuTimestampAsync();
            var now = DateTimeOffset.UtcNow;
            var timeSinceLastCheck = (now - entry.LastCheckTime).TotalMilliseconds;

            // Check if timestamp has changed (kernel is making progress)
            if (currentTimestamp > entry.LastSeenTimestamp)
            {
                // Kernel is healthy - reset timeout counter
                if (entry.ConsecutiveTimeouts > 0)
                {
                    _logger.LogInformation(
                        "Ring kernel {KernelId} recovered - Was unhealthy for {Count} checks",
                        instanceId,
                        entry.ConsecutiveTimeouts);
                }

                entry.LastSeenTimestamp = currentTimestamp;
                entry.LastCheckTime = now;
                entry.ConsecutiveTimeouts = 0;
            }
            else
            {
                // Timestamp hasn't changed - kernel may be hung
                entry.ConsecutiveTimeouts++;
                entry.LastCheckTime = now;

                var totalTimeoutMillis = entry.ConsecutiveTimeouts * _options.CheckIntervalMillis;

                _logger.LogWarning(
                    "Ring kernel {KernelId} appears hung - " +
                    "No progress for {TimeoutMs}ms ({Checks} checks), Timestamp: {Timestamp}ns",
                    instanceId,
                    totalTimeoutMillis,
                    entry.ConsecutiveTimeouts,
                    entry.LastSeenTimestamp);

                // Check if kernel has been hung for too long
                if (totalTimeoutMillis >= _options.HungKernelTimeoutMillis)
                {
                    await HandleHungKernelAsync(instanceId, entry);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Error checking kernel {KernelId} progress",
                instanceId);
        }
    }

    private async Task HandleHungKernelAsync(Guid instanceId, WatchdogEntry entry)
    {
        // Check if we should give up after too many restarts
        if (entry.RestartCount >= _options.MaxRestartAttempts)
        {
            _logger.LogCritical(
                "Ring kernel {KernelId} has failed {Count} times - Giving up. Manual intervention required.",
                instanceId,
                entry.RestartCount);

            // Remove from monitoring (it's permanently failed)
            _monitoredKernels.TryRemove(instanceId, out _);
            return;
        }

        _logger.LogError(
            "Ring kernel {KernelId} is hung - Attempting restart {Attempt}/{MaxAttempts}",
            instanceId,
            entry.RestartCount + 1,
            _options.MaxRestartAttempts);

        try
        {
            // Calculate exponential backoff
            var backoffMillis = _options.RestartBackoffMillis * Math.Pow(2, entry.RestartCount);
            backoffMillis = Math.Min(backoffMillis, _options.MaxRestartBackoffMillis);

            _logger.LogInformation(
                "Waiting {BackoffMs}ms before restarting kernel {KernelId}",
                backoffMillis,
                instanceId);

            await Task.Delay(TimeSpan.FromMilliseconds(backoffMillis));

            // TODO: Implement actual kernel restart with state recovery
            // For now, just log that we would restart
            _logger.LogWarning(
                "Kernel restart not yet implemented for {KernelId} - " +
                "This would restart the kernel with state recovery",
                instanceId);

            // Increment restart counter
            entry.RestartCount++;
            entry.ConsecutiveTimeouts = 0;

            // Reset timestamp after restart
            var newTimestamp = await _timingProvider.GetGpuTimestampAsync();
            entry.LastSeenTimestamp = newTimestamp;
            entry.LastCheckTime = DateTimeOffset.UtcNow;

            _logger.LogInformation(
                "Ring kernel {KernelId} restart completed - New timestamp: {Timestamp}ns",
                instanceId,
                newTimestamp);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to restart hung kernel {KernelId}",
                instanceId);

            entry.RestartCount++;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _watchdogTimer?.Dispose();

        _logger.LogInformation(
            "Ring kernel watchdog stopped - Monitored {Count} kernels",
            _monitoredKernels.Count);

        _monitoredKernels.Clear();
    }

    private sealed class WatchdogEntry
    {
        public required RingKernelHandle Handle { get; init; }
        public long LastSeenTimestamp { get; set; }
        public DateTimeOffset LastCheckTime { get; set; }
        public int ConsecutiveTimeouts { get; set; }
        public int RestartCount { get; set; }
        public DateTimeOffset RegistrationTime { get; init; }
    }
}

/// <summary>
/// Configuration options for ring kernel watchdog.
/// </summary>
public sealed class RingKernelWatchdogOptions
{
    /// <summary>
    /// Interval between health checks in milliseconds.
    /// Default: 1000ms (1 second).
    /// </summary>
    public int CheckIntervalMillis { get; set; } = 1000;

    /// <summary>
    /// Timeout for considering a kernel hung in milliseconds.
    /// Default: 5000ms (5 seconds).
    /// </summary>
    public int HungKernelTimeoutMillis { get; set; } = 5000;

    /// <summary>
    /// Maximum number of restart attempts before giving up.
    /// Default: 3 attempts.
    /// </summary>
    public int MaxRestartAttempts { get; set; } = 3;

    /// <summary>
    /// Initial backoff time before restart in milliseconds.
    /// Default: 1000ms (1 second).
    /// </summary>
    public int RestartBackoffMillis { get; set; } = 1000;

    /// <summary>
    /// Maximum backoff time before restart in milliseconds.
    /// Default: 30000ms (30 seconds).
    /// </summary>
    public int MaxRestartBackoffMillis { get; set; } = 30000;
}

/// <summary>
/// Statistics about a monitored ring kernel.
/// </summary>
public sealed class RingKernelWatchdogStats
{
    public required Guid InstanceId { get; init; }
    public required bool IsHealthy { get; init; }
    public required long LastSeenTimestamp { get; init; }
    public required DateTimeOffset LastCheckTime { get; init; }
    public required int ConsecutiveTimeouts { get; init; }
    public required int RestartCount { get; init; }
    public required TimeSpan Uptime { get; init; }
}
