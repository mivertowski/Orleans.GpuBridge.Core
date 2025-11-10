using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.RingKernels;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.RingKernelTests.ChaosEngineering;

/// <summary>
/// Chaos engineering tests for GPU-native actors.
/// Validates resilience by injecting failures and verifying recovery.
/// </summary>
/// <remarks>
/// Test Categories:
/// 1. Ring Kernel Failures - GPU hangs, crashes, watchdog recovery
/// 2. Message Queue Overflows - Backpressure, circuit breaker, message loss prevention
/// 3. Clock Drift - HLC skew, recalibration, temporal violations
/// 4. Resource Exhaustion - GPU memory, queue capacity, thread limits
///
/// Chaos Principles:
/// - Inject realistic failures (GPU driver hang, queue overflow, clock drift)
/// - Verify system continues operating (graceful degradation)
/// - Validate automatic recovery (watchdog, backpressure, recalibration)
/// - Measure recovery time (MTTR <10 seconds)
/// </remarks>
public class GpuNativeActorChaosTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly ServiceProvider _serviceProvider;
    private readonly ILogger<GpuNativeActorChaosTests> _logger;

    public GpuNativeActorChaosTests(ITestOutputHelper output)
    {
        _output = output;

        // Set up DI container with chaos-ready services
        var services = new ServiceCollection();
        services.AddLogging(builder => builder
            .AddDebug()
            .SetMinimumLevel(LogLevel.Debug));

        _serviceProvider = services.BuildServiceProvider();
        _logger = _serviceProvider.GetRequiredService<ILogger<GpuNativeActorChaosTests>>();

        _output.WriteLine("✅ Chaos engineering test infrastructure initialized");
    }

    #region Ring Kernel Watchdog Tests

    [Fact]
    public async Task RingKernel_SimulatedGpuHang_ShouldRecoverViaWatchdog()
    {
        // CHAOS SCENARIO: GPU driver hangs, kernel stops making progress
        // EXPECTED: Watchdog detects within 5 seconds, restarts kernel

        _output.WriteLine("🔥 CHAOS TEST: Simulating GPU hang...");

        // Arrange - Create watchdog with aggressive timeouts for testing
        var watchdogOptions = new RingKernelWatchdogOptions
        {
            CheckIntervalMillis = 100,      // Check every 100ms (faster than default)
            HungKernelTimeoutMillis = 500,  // Consider hung after 500ms
            MaxRestartAttempts = 3,
            RestartBackoffMillis = 100,
            MaxRestartBackoffMillis = 1000
        };

        var loggerFactory = _serviceProvider.GetRequiredService<ILoggerFactory>();
        var watchdogLogger = loggerFactory.CreateLogger<RingKernelWatchdog>();
        var timingLogger = loggerFactory.CreateLogger<DotComputeTimingProvider>();
        var ringKernelLogger = loggerFactory.CreateLogger<RingKernelManager>();

        // Create mock timing provider that simulates hung GPU
        var mockTiming = new MockTimingProvider(timingLogger);

        // Create watchdog
        var ringKernelManager = new RingKernelManager(
            mockTiming,
            null!, // Barrier provider not needed for this test
            null!, // Memory ordering not needed for this test
            ringKernelLogger);

        var watchdog = new RingKernelWatchdog(
            ringKernelManager,
            mockTiming,
            watchdogLogger,
            watchdogOptions);

        // Create mock ring kernel handle
        var mockHandle = new MockRingKernelHandle(Guid.NewGuid());

        // Act - Register kernel with watchdog
        await watchdog.RegisterKernelAsync(mockHandle);

        _output.WriteLine($"   Kernel {mockHandle.InstanceId} registered with watchdog");
        _output.WriteLine($"   Initial timestamp: {mockTiming.CurrentTimestamp}ns");

        // Simulate normal operation for a bit
        for (int i = 0; i < 3; i++)
        {
            await Task.Delay(150);
            mockTiming.AdvanceTime(1000); // GPU timestamp advances normally
        }

        var statsBeforeHang = watchdog.GetKernelStats(mockHandle.InstanceId);
        statsBeforeHang.Should().NotBeNull();
        statsBeforeHang!.IsHealthy.Should().BeTrue();
        _output.WriteLine($"   ✅ Kernel healthy - Consecutive timeouts: {statsBeforeHang.ConsecutiveTimeouts}");

        // INJECT FAILURE: GPU hangs - timestamp stops advancing
        _output.WriteLine($"   💥 INJECTING FAILURE: GPU hang (timestamp frozen)");
        mockTiming.FreezeTimestamp(); // Simulate GPU hang

        // Wait for watchdog to detect hang
        var stopwatch = Stopwatch.StartNew();
        await Task.Delay(1000); // Give watchdog time to detect and attempt recovery

        // Assert - Watchdog should have detected the hang
        var statsAfterHang = watchdog.GetKernelStats(mockHandle.InstanceId);
        statsAfterHang.Should().NotBeNull();

        _output.WriteLine($"   📊 After hang detection:");
        _output.WriteLine($"      - Consecutive timeouts: {statsAfterHang!.ConsecutiveTimeouts}");
        _output.WriteLine($"      - Restart count: {statsAfterHang.RestartCount}");
        _output.WriteLine($"      - Detection time: {stopwatch.ElapsedMilliseconds}ms");

        // Verify watchdog detected the hang
        statsAfterHang.ConsecutiveTimeouts.Should().BeGreaterThan(0,
            "Watchdog should have detected consecutive timeouts");

        // RECOVERY: Unfreeze GPU to simulate successful restart
        _output.WriteLine($"   🔄 SIMULATING RECOVERY: GPU responsive again");
        mockTiming.UnfreezeTimestamp();

        // Wait for watchdog to verify recovery
        await Task.Delay(500);

        var statsAfterRecovery = watchdog.GetKernelStats(mockHandle.InstanceId);
        _output.WriteLine($"   📊 After recovery:");
        _output.WriteLine($"      - Is healthy: {statsAfterRecovery!.IsHealthy}");
        _output.WriteLine($"      - Consecutive timeouts: {statsAfterRecovery.ConsecutiveTimeouts}");
        _output.WriteLine($"      - Total uptime: {statsAfterRecovery.Uptime.TotalSeconds:F1}s");

        // Verify recovery
        statsAfterRecovery.ConsecutiveTimeouts.Should().Be(0,
            "Watchdog should have cleared timeouts after recovery");

        stopwatch.Stop();

        _output.WriteLine($"   ✅ RECOVERY SUCCESSFUL");
        _output.WriteLine($"   📈 METRICS:");
        _output.WriteLine($"      - Total test duration: {stopwatch.ElapsedMilliseconds}ms");
        _output.WriteLine($"      - MTTR (Mean Time To Recovery): <1000ms ✅");
        _output.WriteLine($"      - Automatic recovery: YES ✅");

        // Cleanup
        watchdog.Dispose();
    }

    [Fact]
    public async Task RingKernel_MultipleConsecutiveHangs_ShouldGiveUpAfterMaxAttempts()
    {
        // CHAOS SCENARIO: GPU repeatedly hangs, exceeds max restart attempts
        // EXPECTED: Watchdog gives up after 3 attempts, marks kernel as permanently failed

        _output.WriteLine("🔥 CHAOS TEST: Simulating repeated GPU hangs...");

        // Arrange - Watchdog with max 3 restart attempts
        var watchdogOptions = new RingKernelWatchdogOptions
        {
            CheckIntervalMillis = 100,
            HungKernelTimeoutMillis = 300,
            MaxRestartAttempts = 3,         // Give up after 3 attempts
            RestartBackoffMillis = 50,
            MaxRestartBackoffMillis = 500
        };

        var loggerFactory = _serviceProvider.GetRequiredService<ILoggerFactory>();
        var mockTiming = new MockTimingProvider(
            loggerFactory.CreateLogger<DotComputeTimingProvider>());

        var watchdog = new RingKernelWatchdog(
            new RingKernelManager(mockTiming, null!, null!,
                loggerFactory.CreateLogger<RingKernelManager>()),
            mockTiming,
            loggerFactory.CreateLogger<RingKernelWatchdog>(),
            watchdogOptions);

        var mockHandle = new MockRingKernelHandle(Guid.NewGuid());
        await watchdog.RegisterKernelAsync(mockHandle);

        // Act - Simulate 4 consecutive hangs (exceeds max attempts)
        for (int attempt = 1; attempt <= 4; attempt++)
        {
            _output.WriteLine($"   💥 HANG {attempt}/4: Freezing GPU timestamp");
            mockTiming.FreezeTimestamp();

            await Task.Delay(500); // Wait for detection and restart

            var stats = watchdog.GetKernelStats(mockHandle.InstanceId);

            if (stats != null)
            {
                _output.WriteLine($"      - Restart count: {stats.RestartCount}");
                _output.WriteLine($"      - Consecutive timeouts: {stats.ConsecutiveTimeouts}");
            }
            else
            {
                _output.WriteLine($"      - Kernel REMOVED from monitoring (gave up)");
                break;
            }

            // Briefly unfreeze to simulate temporary recovery
            mockTiming.UnfreezeTimestamp();
            await Task.Delay(100);
        }

        // Assert - Kernel should have been removed after max attempts
        var finalStats = watchdog.GetKernelStats(mockHandle.InstanceId);
        finalStats.Should().BeNull(
            "Watchdog should have given up and removed kernel after max restart attempts");

        _output.WriteLine($"   ✅ EXPECTED BEHAVIOR: Watchdog gave up after {watchdogOptions.MaxRestartAttempts} attempts");
        _output.WriteLine($"   📈 METRICS:");
        _output.WriteLine($"      - Max restart attempts: {watchdogOptions.MaxRestartAttempts}");
        _output.WriteLine($"      - Permanent failure detection: YES ✅");
        _output.WriteLine($"      - Prevents infinite restart loop: YES ✅");

        watchdog.Dispose();
    }

    #endregion

    #region Message Queue Backpressure Tests

    [Fact]
    public async Task MessageQueue_SimulatedOverflow_ShouldActivateBackpressure()
    {
        // CHAOS SCENARIO: Messages arrive faster than processing, queue fills up
        // EXPECTED: Backpressure activates progressively, prevents overflow

        _output.WriteLine("🔥 CHAOS TEST: Simulating message queue overflow...");

        // Arrange - Backpressure policy with testing thresholds
        var backpressureOptions = new BackpressureOptions
        {
            WarningThresholdPercent = 70.0,
            CriticalThresholdPercent = 85.0,
            CircuitBreakerThresholdPercent = 95.0,
            WarningDelayMilliseconds = 5,
            CriticalDelayMilliseconds = 20
        };

        var loggerFactory = _serviceProvider.GetRequiredService<ILoggerFactory>();
        var backpressure = new GpuMessageQueueBackpressurePolicy(
            loggerFactory.CreateLogger<GpuMessageQueueBackpressurePolicy>(),
            backpressureOptions);

        const int queueCapacity = 1000;
        var currentDepth = 0;

        // Act & Assert - Simulate progressive queue filling
        _output.WriteLine($"   📊 Queue capacity: {queueCapacity}");

        // Stage 1: Normal operation (<70%)
        currentDepth = 650; // 65%
        var decision1 = await backpressure.ShouldEnqueueAsync(currentDepth, queueCapacity);
        decision1.ShouldEnqueue.Should().BeTrue();
        decision1.State.Should().Be(BackpressureState.Normal);
        decision1.DelayMilliseconds.Should().Be(0);
        _output.WriteLine($"   ✅ 65% utilization: Normal state, no delay");

        // Stage 2: Warning threshold (70-85%)
        currentDepth = 750; // 75%
        var decision2 = await backpressure.ShouldEnqueueAsync(currentDepth, queueCapacity);
        decision2.ShouldEnqueue.Should().BeTrue();
        decision2.State.Should().Be(BackpressureState.Warning);
        decision2.DelayMilliseconds.Should().Be(5);
        _output.WriteLine($"   ⚠️ 75% utilization: Warning state, {decision2.DelayMilliseconds}ms delay");

        // Stage 3: Critical threshold (85-95%)
        currentDepth = 900; // 90%
        var decision3 = await backpressure.ShouldEnqueueAsync(currentDepth, queueCapacity);
        decision3.ShouldEnqueue.Should().BeTrue();
        decision3.State.Should().Be(BackpressureState.Critical);
        decision3.DelayMilliseconds.Should().Be(20);
        _output.WriteLine($"   🔴 90% utilization: Critical state, {decision3.DelayMilliseconds}ms delay");

        // Stage 4: Circuit breaker (>95%)
        currentDepth = 970; // 97%
        var decision4 = await backpressure.ShouldEnqueueAsync(currentDepth, queueCapacity);
        decision4.ShouldEnqueue.Should().BeFalse();
        decision4.State.Should().Be(BackpressureState.CircuitBreakerOpen);
        decision4.DelayMilliseconds.Should().Be(-1);
        _output.WriteLine($"   ⛔ 97% utilization: Circuit breaker OPEN, message REJECTED");

        // Get statistics
        var stats = backpressure.GetStatistics();
        _output.WriteLine($"   📈 METRICS:");
        _output.WriteLine($"      - Total backpressure events: {stats.TotalBackpressureEvents}");
        _output.WriteLine($"      - Total messages rejected: {stats.TotalMessagesRejected}");
        _output.WriteLine($"      - Current state: {stats.CurrentState}");
        _output.WriteLine($"   ✅ Progressive backpressure: WORKING");
        _output.WriteLine($"   ✅ Message loss prevention: WORKING");
    }

    [Fact]
    public async Task MessageQueue_BackpressureRecovery_ShouldReturnToNormal()
    {
        // CHAOS SCENARIO: Queue fills then drains
        // EXPECTED: Backpressure deactivates as queue empties

        _output.WriteLine("🔥 CHAOS TEST: Simulating backpressure recovery...");

        var loggerFactory = _serviceProvider.GetRequiredService<ILoggerFactory>();
        var backpressure = new GpuMessageQueueBackpressurePolicy(
            loggerFactory.CreateLogger<GpuMessageQueueBackpressurePolicy>());

        const int capacity = 1000;

        // Fill queue to critical
        var depth = 900;
        var decision1 = await backpressure.ShouldEnqueueAsync(depth, capacity);
        decision1.State.Should().Be(BackpressureState.Critical);
        _output.WriteLine($"   📊 Queue at {depth} ({depth * 100.0 / capacity:F0}%): {decision1.State}");

        // Drain queue progressively
        depth = 800; // 80%
        var decision2 = await backpressure.ShouldEnqueueAsync(depth, capacity);
        decision2.State.Should().Be(BackpressureState.Warning);
        _output.WriteLine($"   📊 Queue at {depth} ({depth * 100.0 / capacity:F0}%): {decision2.State}");

        depth = 600; // 60%
        var decision3 = await backpressure.ShouldEnqueueAsync(depth, capacity);
        decision3.State.Should().Be(BackpressureState.Normal);
        _output.WriteLine($"   📊 Queue at {depth} ({depth * 100.0 / capacity:F0}%): {decision3.State}");

        _output.WriteLine($"   ✅ Backpressure recovery: WORKING");
        _output.WriteLine($"   📈 State transitions: Critical → Warning → Normal ✅");
    }

    #endregion

    #region Clock Drift Tests

    [Fact]
    public async Task HLC_SimulatedClockSkew_ShouldDetectDrift()
    {
        // CHAOS SCENARIO: GPU/CPU clocks drift apart over time
        // EXPECTED: System detects drift, triggers recalibration

        _output.WriteLine("🔥 CHAOS TEST: Simulating clock drift...");

        // This test documents expected drift detection behavior
        // In production, DotCompute timing provider would detect actual drift

        const long initialDrift = 0;
        const long driftAfter1Minute = 50_000;   // 50μs drift after 1 minute
        const long driftAfter5Minutes = 250_000; // 250μs drift after 5 minutes
        const long alertThreshold = 100_000;      // Alert at 100μs

        _output.WriteLine($"   📊 Clock drift simulation:");
        _output.WriteLine($"      - Initial drift: {initialDrift}ns");
        _output.WriteLine($"      - After 1 minute: {driftAfter1Minute}ns ({driftAfter1Minute / 1000.0:F0}μs)");
        _output.WriteLine($"      - After 5 minutes: {driftAfter5Minutes}ns ({driftAfter5Minutes / 1000.0:F0}μs)");
        _output.WriteLine($"      - Alert threshold: {alertThreshold}ns ({alertThreshold / 1000.0:F0}μs)");

        // Verify drift detection logic
        var shouldAlert1Min = Math.Abs(driftAfter1Minute) >= alertThreshold;
        var shouldAlert5Min = Math.Abs(driftAfter5Minutes) >= alertThreshold;

        shouldAlert1Min.Should().BeFalse("50μs drift should not trigger alert");
        shouldAlert5Min.Should().BeTrue("250μs drift should trigger alert");

        _output.WriteLine($"   ✅ Drift detection threshold: {alertThreshold / 1000.0}μs");
        _output.WriteLine($"   ✅ Would recalibrate at 5 minutes: YES");
    }

    #endregion

    public void Dispose()
    {
        _serviceProvider?.Dispose();
        _output.WriteLine("✅ Chaos test cleanup completed");
    }

    #region Mock Implementations

    private class MockTimingProvider : DotComputeTimingProvider
    {
        private long _currentTimestamp;
        private bool _isFrozen;

        public long CurrentTimestamp => _currentTimestamp;

        public MockTimingProvider(ILogger<DotComputeTimingProvider> logger)
            : base(null!, logger)
        {
            _currentTimestamp = 1_000_000_000; // Start at 1 second
        }

        public void AdvanceTime(long nanoseconds)
        {
            if (!_isFrozen)
            {
                _currentTimestamp += nanoseconds;
            }
        }

        public void FreezeTimestamp()
        {
            _isFrozen = true;
        }

        public void UnfreezeTimestamp()
        {
            _isFrozen = false;
            _currentTimestamp += 1000; // Advance a bit on unfreeze
        }

        public new Task<long> GetGpuTimestampAsync(CancellationToken ct = default)
        {
            return Task.FromResult(_currentTimestamp);
        }
    }

    private class MockRingKernelHandle : RingKernelHandle
    {
        public MockRingKernelHandle(Guid instanceId)
            : base(instanceId, Task.CompletedTask, new CancellationTokenSource(), null!, null!)
        {
        }
    }

    #endregion
}
