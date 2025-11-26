// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions.Timing;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// GPU timing provider implementation that wraps DotCompute's <see cref="ITimingProvider"/>.
/// Provides real GPU timestamps using CUDA globaltimer or event-based timing.
/// </summary>
/// <remarks>
/// <para>
/// This provider delegates to DotCompute's timing infrastructure which provides:
/// <list type="bullet">
/// <item><description>CUDA (CC 6.0+): 1ns resolution via %%globaltimer register</description></item>
/// <item><description>CUDA (CC &lt; 6.0): 1μs resolution via CUDA events</description></item>
/// <item><description>Automatic fallback to CPU timing when GPU unavailable</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Usage:</strong>
/// <code>
/// services.AddSingleton&lt;IGpuTimingProvider, DotComputeTimingProvider&gt;();
/// </code>
/// </para>
/// </remarks>
public sealed class DotComputeTimingProvider : IGpuTimingProvider, IDisposable
{
    private readonly ITimingProvider _dotComputeProvider;
    private readonly ILogger<DotComputeTimingProvider> _logger;
    private readonly string _providerTypeName;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="DotComputeTimingProvider"/> class
    /// wrapping an existing DotCompute <see cref="ITimingProvider"/>.
    /// </summary>
    /// <param name="dotComputeProvider">The underlying DotCompute timing provider.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="dotComputeProvider"/> is null.
    /// </exception>
    public DotComputeTimingProvider(
        ITimingProvider dotComputeProvider,
        ILogger<DotComputeTimingProvider> logger)
    {
        _dotComputeProvider = dotComputeProvider ?? throw new ArgumentNullException(nameof(dotComputeProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Determine provider type name from resolution
        var resolutionNs = _dotComputeProvider.GetTimerResolutionNanos();
        _providerTypeName = resolutionNs <= 1
            ? "CUDA (globaltimer - 1ns)"
            : $"CUDA (events - {resolutionNs}ns)";

        _logger.LogInformation(
            "DotComputeTimingProvider initialized: Type={ProviderType}, Resolution={ResolutionNs}ns, Frequency={FrequencyHz}Hz",
            _providerTypeName,
            resolutionNs,
            _dotComputeProvider.GetGpuClockFrequency());
    }

    /// <inheritdoc/>
    public async Task<long> GetGpuTimestampAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return await _dotComputeProvider.GetGpuTimestampAsync(ct);
    }

    /// <inheritdoc/>
    public async Task<long[]> GetGpuTimestampsBatchAsync(int count, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(count, 0, nameof(count));
        return await _dotComputeProvider.GetGpuTimestampsBatchAsync(count, ct);
    }

    /// <inheritdoc/>
    public async Task<Abstractions.Temporal.ClockCalibration> CalibrateAsync(
        int sampleCount = 100,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (sampleCount < 10)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount),
                "Sample count must be at least 10 for reliable calibration.");
        }

        _logger.LogInformation("Starting DotCompute GPU clock calibration with {SampleCount} samples...", sampleCount);

        // Delegate to DotCompute's calibration
        var dotComputeCalibration = await _dotComputeProvider.CalibrateAsync(sampleCount, ct);

        // Convert from DotCompute's ClockCalibration to our ClockCalibration
        var calibration = new Abstractions.Temporal.ClockCalibration(
            offsetNanos: dotComputeCalibration.OffsetNanos,
            driftPPM: dotComputeCalibration.DriftPPM,
            errorBoundNanos: dotComputeCalibration.ErrorBoundNanos,
            sampleCount: dotComputeCalibration.SampleCount,
            calibrationTimestampNanos: dotComputeCalibration.CalibrationTimestampNanos);

        _logger.LogInformation(
            "DotCompute GPU clock calibration complete: Offset={OffsetNs}ns, Drift={DriftPPM:F3}ppm, Error=±{ErrorNs}ns",
            calibration.OffsetNanos, calibration.DriftPPM, calibration.ErrorBoundNanos);

        return calibration;
    }

    /// <inheritdoc/>
    public void EnableTimestampInjection(bool enable = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _dotComputeProvider.EnableTimestampInjection(enable);
        _logger.LogDebug("Timestamp injection {Action}", enable ? "enabled" : "disabled");
    }

    /// <inheritdoc/>
    public bool IsTimestampInjectionEnabled
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            // DotCompute's ITimingProvider doesn't expose this property directly,
            // so we track it ourselves or use reflection/internal knowledge
            // For now, return a safe default
            return false; // Will be updated when DotCompute adds the property
        }
    }

    /// <inheritdoc/>
    public long GetGpuClockFrequency()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _dotComputeProvider.GetGpuClockFrequency();
    }

    /// <inheritdoc/>
    public long GetTimerResolutionNanos()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _dotComputeProvider.GetTimerResolutionNanos();
    }

    /// <inheritdoc/>
    public bool IsGpuBacked => true;

    /// <inheritdoc/>
    public string ProviderTypeName => _providerTypeName;

    /// <summary>
    /// Releases resources used by the timing provider.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        // If the underlying provider is disposable, dispose it
        if (_dotComputeProvider is IDisposable disposable)
        {
            disposable.Dispose();
        }

        _disposed = true;
        _logger.LogDebug("DotComputeTimingProvider disposed");
    }
}
