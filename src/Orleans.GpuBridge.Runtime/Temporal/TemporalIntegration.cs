// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Provides GPU temporal integration capabilities for Orleans actors.
/// </summary>
/// <remarks>
/// <para>
/// This implementation abstracts over DotCompute's temporal features:
/// <list type="bullet">
/// <item><description>GPU-side timestamp injection via [Kernel] attributes</description></item>
/// <item><description>Device-wide barriers via Cooperative Groups</description></item>
/// <item><description>Memory ordering via fence primitives</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Performance Characteristics:</strong>
/// <list type="bullet">
/// <item><description>GPU timestamp resolution: ~20ns (CUDA %%globaltimer)</description></item>
/// <item><description>Barrier synchronization: &lt;1μs (device-wide)</description></item>
/// <item><description>Memory fence overhead: &lt;50ns (device scope)</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class TemporalIntegration : ITemporalIntegration, IDisposable
{
    private readonly ILogger<TemporalIntegration> _logger;
    private readonly IGpuTimingProvider _timingProvider;
    private readonly GpuClockCalibrator _calibrator;
    private readonly ITemporalBarrierManager _barrierManager;
    private readonly ICausalMemoryOrdering _memoryOrdering;

    private TemporalKernelOptions _currentOptions = new();
    private ClockCalibration? _currentCalibration;
    private bool _disposed;

    // Feature detection flags
    private readonly bool _isGpuTimingAvailable;
    private readonly bool _areBarriersSupported;
    private readonly bool _isMemoryOrderingSupported;

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalIntegration"/> class.
    /// </summary>
    /// <param name="timingProvider">GPU timing provider for timestamp operations.</param>
    /// <param name="calibrator">Clock calibrator for GPU-CPU time synchronization.</param>
    /// <param name="barrierManager">Manager for device-wide barriers.</param>
    /// <param name="memoryOrdering">Memory ordering primitives.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public TemporalIntegration(
        IGpuTimingProvider timingProvider,
        GpuClockCalibrator calibrator,
        ITemporalBarrierManager barrierManager,
        ICausalMemoryOrdering memoryOrdering,
        ILogger<TemporalIntegration> logger)
    {
        _timingProvider = timingProvider ?? throw new ArgumentNullException(nameof(timingProvider));
        _calibrator = calibrator ?? throw new ArgumentNullException(nameof(calibrator));
        _barrierManager = barrierManager ?? throw new ArgumentNullException(nameof(barrierManager));
        _memoryOrdering = memoryOrdering ?? throw new ArgumentNullException(nameof(memoryOrdering));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Detect available features
        _isGpuTimingAvailable = _timingProvider.IsGpuBacked;
        _areBarriersSupported = DetectBarrierSupport();
        _isMemoryOrderingSupported = DetectMemoryOrderingSupport();

        _logger.LogInformation(
            "TemporalIntegration initialized: GPU={IsGpu}, Barriers={Barriers}, MemoryOrdering={MemOrder}",
            _isGpuTimingAvailable,
            _areBarriersSupported,
            _isMemoryOrderingSupported);
    }

    /// <summary>
    /// Initializes a new instance with minimal dependencies (CPU fallback mode).
    /// </summary>
    /// <param name="timingProvider">Timing provider (can be CPU-based).</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public TemporalIntegration(
        IGpuTimingProvider timingProvider,
        ILogger<TemporalIntegration> logger)
    {
        _timingProvider = timingProvider ?? throw new ArgumentNullException(nameof(timingProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Create CPU-based fallbacks
        var calibratorLogger = new NullLogger<GpuClockCalibrator>();
        _calibrator = new GpuClockCalibrator(_timingProvider, calibratorLogger);

        var barrierLogger = new NullLogger<TemporalBarrierManager>();
        _barrierManager = new TemporalBarrierManager(barrierLogger);

        var memOrderLogger = new NullLogger<CausalMemoryOrdering>();
        _memoryOrdering = new CausalMemoryOrdering(memOrderLogger);

        // Detect available features
        _isGpuTimingAvailable = _timingProvider.IsGpuBacked;
        _areBarriersSupported = DetectBarrierSupport();
        _isMemoryOrderingSupported = DetectMemoryOrderingSupport();

        _logger.LogInformation(
            "TemporalIntegration initialized (minimal): GPU={IsGpu}, Barriers={Barriers}, MemoryOrdering={MemOrder}",
            _isGpuTimingAvailable,
            _areBarriersSupported,
            _isMemoryOrderingSupported);
    }

    /// <inheritdoc/>
    public async Task ConfigureTemporalKernelAsync(
        TemporalKernelOptions options,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        ObjectDisposedException.ThrowIf(_disposed, this);

        _logger.LogInformation(
            "Configuring temporal kernel: Timestamps={Timestamps}, Barriers={Barriers}, " +
            "BarrierScope={Scope}, MemoryOrdering={MemOrder}, Fences={Fences}",
            options.EnableTimestamps,
            options.EnableBarriers,
            options.BarrierScope,
            options.MemoryOrdering,
            options.EnableFences);

        // Enable/disable timestamp injection on the timing provider
        _timingProvider.EnableTimestampInjection(options.EnableTimestamps);

        // Configure memory ordering mode
        if (options.EnableFences)
        {
            _memoryOrdering.SetMode(options.MemoryOrdering);
        }

        // Validate barrier configuration
        if (options.EnableBarriers && !_areBarriersSupported)
        {
            _logger.LogWarning(
                "Barriers requested but not supported on this device. " +
                "Falling back to CPU synchronization.");
        }

        // Validate barrier scope
        if (options.EnableBarriers && options.BarrierScope == BarrierScope.System && !_isGpuTimingAvailable)
        {
            _logger.LogWarning(
                "System-scope barriers require GPU timing. " +
                "Downgrading to Device scope.");
            options = new TemporalKernelOptions
            {
                EnableTimestamps = options.EnableTimestamps,
                EnableBarriers = options.EnableBarriers,
                BarrierScope = BarrierScope.Device,
                MemoryOrdering = options.MemoryOrdering,
                BarrierTimeoutMs = options.BarrierTimeoutMs,
                EnableFences = options.EnableFences,
                FenceScope = options.FenceScope
            };
        }

        _currentOptions = options;

        // Perform initial calibration if timestamps are enabled
        if (options.EnableTimestamps && _currentCalibration == null)
        {
            _currentCalibration = await CalibrateGpuClockAsync(1000, ct);
        }

        _logger.LogDebug("Temporal kernel configuration complete");
    }

    /// <inheritdoc/>
    public async Task<ClockCalibration> CalibrateGpuClockAsync(
        int sampleCount = 1000,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (sampleCount < 10)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sampleCount),
                "Sample count must be at least 10 for reliable calibration.");
        }

        _logger.LogInformation(
            "Starting GPU clock calibration with {SampleCount} samples...",
            sampleCount);

        var calibration = await _calibrator.CalibrateAsync(sampleCount, ct);
        _currentCalibration = calibration;

        _logger.LogInformation(
            "GPU clock calibration complete: Offset={OffsetNs}ns, Drift={DriftPPM:F3}ppm, " +
            "Error=±{ErrorNs}ns (Provider: {Provider})",
            calibration.OffsetNanos,
            calibration.DriftPPM,
            calibration.ErrorBoundNanos,
            _timingProvider.ProviderTypeName);

        return calibration;
    }

    /// <inheritdoc/>
    public async Task<long> GetGpuTimestampAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return await _timingProvider.GetGpuTimestampAsync(ct);
    }

    /// <inheritdoc/>
    public ClockCalibration? CurrentCalibration => _currentCalibration;

    /// <inheritdoc/>
    public IGpuTimingProvider TimingProvider => _timingProvider;

    /// <inheritdoc/>
    public bool IsGpuTimingAvailable => _isGpuTimingAvailable;

    /// <inheritdoc/>
    public bool AreBarriersSupported => _areBarriersSupported;

    /// <inheritdoc/>
    public bool IsMemoryOrderingSupported => _isMemoryOrderingSupported;

    /// <summary>
    /// Gets the barrier manager for device-wide synchronization.
    /// </summary>
    public ITemporalBarrierManager BarrierManager => _barrierManager;

    /// <summary>
    /// Gets the memory ordering controller.
    /// </summary>
    public ICausalMemoryOrdering MemoryOrdering => _memoryOrdering;

    /// <summary>
    /// Gets the current temporal kernel options.
    /// </summary>
    public TemporalKernelOptions CurrentOptions => _currentOptions;

    /// <summary>
    /// Creates a device-wide barrier for synchronization.
    /// </summary>
    /// <param name="scope">Barrier scope (ThreadBlock, Device, Grid, System).</param>
    /// <param name="timeoutMs">Timeout in milliseconds.</param>
    /// <returns>The created barrier.</returns>
    public ITemporalBarrier CreateBarrier(BarrierScope scope = BarrierScope.Device, int timeoutMs = 5000)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_areBarriersSupported && scope > BarrierScope.ThreadBlock)
        {
            _logger.LogWarning(
                "Creating barrier with scope {Scope} but barriers not fully supported. " +
                "Using CPU synchronization fallback.",
                scope);
        }

        return _barrierManager.CreateBarrier(scope, timeoutMs);
    }

    /// <summary>
    /// Executes a device-wide barrier synchronization.
    /// </summary>
    /// <param name="barrier">Barrier to synchronize on.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task SynchronizeBarrierAsync(ITemporalBarrier barrier, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(barrier);

        await _barrierManager.SynchronizeAsync(barrier, ct);
    }

    /// <summary>
    /// Inserts a memory fence with the configured scope.
    /// </summary>
    /// <returns>Fence identifier for tracking.</returns>
    public long InsertMemoryFence()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_currentOptions.EnableFences)
        {
            _logger.LogDebug("Memory fences disabled, skipping fence insertion");
            return 0;
        }

        return _memoryOrdering.InsertFence(_currentOptions.FenceScope);
    }

    /// <summary>
    /// Applies acquire semantics for memory ordering.
    /// </summary>
    public void AcquireMemorySemantics()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _memoryOrdering.AcquireSemantics();
    }

    /// <summary>
    /// Applies release semantics for memory ordering.
    /// </summary>
    public void ReleaseMemorySemantics()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _memoryOrdering.ReleaseSemantics();
    }

    /// <summary>
    /// Gets comprehensive temporal integration statistics.
    /// </summary>
    public TemporalIntegrationStatistics GetStatistics()
    {
        var barrierStats = _barrierManager.GetStatistics();
        var memOrderStats = _memoryOrdering.GetStatistics();

        return new TemporalIntegrationStatistics
        {
            IsGpuBacked = _isGpuTimingAvailable,
            ProviderTypeName = _timingProvider.ProviderTypeName,
            TimerResolutionNanos = _timingProvider.GetTimerResolutionNanos(),
            ClockFrequencyHz = _timingProvider.GetGpuClockFrequency(),
            CurrentCalibration = _currentCalibration,
            BarrierStatistics = barrierStats,
            MemoryOrderingStatistics = memOrderStats,
            ConfiguredOptions = _currentOptions
        };
    }

    /// <summary>
    /// Detects whether device-wide barriers are supported.
    /// </summary>
    private bool DetectBarrierSupport()
    {
        // In a full implementation, this would query the GPU device capabilities
        // For now, barriers are simulated in CPU mode
        // When DotCompute 0.4.2-rc2 is integrated, this will check CooperativeGroups support

        if (_isGpuTimingAvailable)
        {
            // GPU available - assume barrier support (CUDA Compute Capability 6.0+)
            _logger.LogDebug("GPU detected, assuming barrier support");
            return true;
        }

        // CPU fallback - barriers are simulated
        _logger.LogDebug("No GPU detected, using simulated barriers");
        return false;
    }

    /// <summary>
    /// Detects whether memory ordering primitives are supported.
    /// </summary>
    private bool DetectMemoryOrderingSupport()
    {
        // Memory ordering is available on all platforms
        // GPU memory fences use __threadfence variants
        // CPU uses Thread.MemoryBarrier
        return true;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _calibrator.Dispose();

        _logger.LogDebug("TemporalIntegration disposed");
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"TemporalIntegration(GPU={_isGpuTimingAvailable}, " +
               $"Provider={_timingProvider.ProviderTypeName}, " +
               $"Barriers={_areBarriersSupported})";
    }
}

/// <summary>
/// Comprehensive statistics about temporal integration.
/// </summary>
public sealed record TemporalIntegrationStatistics
{
    /// <summary>
    /// Whether the integration is backed by real GPU hardware.
    /// </summary>
    public bool IsGpuBacked { get; init; }

    /// <summary>
    /// Name of the timing provider.
    /// </summary>
    public required string ProviderTypeName { get; init; }

    /// <summary>
    /// Timer resolution in nanoseconds.
    /// </summary>
    public long TimerResolutionNanos { get; init; }

    /// <summary>
    /// GPU clock frequency in Hz.
    /// </summary>
    public long ClockFrequencyHz { get; init; }

    /// <summary>
    /// Current clock calibration (null if not calibrated).
    /// </summary>
    public ClockCalibration? CurrentCalibration { get; init; }

    /// <summary>
    /// Barrier operation statistics.
    /// </summary>
    public required BarrierStatistics BarrierStatistics { get; init; }

    /// <summary>
    /// Memory ordering statistics.
    /// </summary>
    public required MemoryOrderingStatistics MemoryOrderingStatistics { get; init; }

    /// <summary>
    /// Currently configured options.
    /// </summary>
    public required TemporalKernelOptions ConfiguredOptions { get; init; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"TemporalStats(GPU={IsGpuBacked}, Resolution={TimerResolutionNanos}ns, " +
               $"Barriers={BarrierStatistics.TotalBarriersCreated}, " +
               $"Fences={MemoryOrderingStatistics.TotalFencesInserted})";
    }
}

/// <summary>
/// Null logger implementation for minimal dependency constructors.
/// </summary>
internal sealed class NullLogger<T> : ILogger<T>
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(LogLevel logLevel) => false;
    public void Log<TState>(
        LogLevel logLevel,
        EventId eventId,
        TState state,
        Exception? exception,
        Func<TState, Exception?, string> formatter)
    { }
}
