using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using FluentAssertions.Execution;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Tests.RC2.TestingFramework;

/// <summary>
/// Helper methods and utilities for GPU Bridge testing.
/// Provides assertion helpers, retry logic, performance measurement, and diagnostic utilities.
/// </summary>
/// <remarks>
/// These helpers are designed to make GPU testing more robust by handling common
/// scenarios like hardware flakiness, timing-sensitive operations, and result validation.
/// All methods are thread-safe and can be used in parallel test scenarios.
/// </remarks>
public static class GpuTestHelpers
{
    /// <summary>
    /// Default timeout for asynchronous operations in tests.
    /// </summary>
    public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Default retry count for flaky operations.
    /// </summary>
    public const int DefaultRetryCount = 3;

    /// <summary>
    /// Default delay between retry attempts.
    /// </summary>
    public static readonly TimeSpan DefaultRetryDelay = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Tolerance for float comparisons in assertions.
    /// </summary>
    public const float DefaultFloatTolerance = 1e-6f;

    /// <summary>
    /// Tolerance for double comparisons in assertions.
    /// </summary>
    public const double DefaultDoubleTolerance = 1e-10;

    #region Assertion Helpers

    /// <summary>
    /// Asserts that two float arrays are approximately equal within specified tolerance.
    /// </summary>
    /// <param name="actual">The actual array.</param>
    /// <param name="expected">The expected array.</param>
    /// <param name="tolerance">Maximum allowed difference per element.</param>
    /// <param name="because">Reason for the assertion.</param>
    public static void ShouldBeApproximately(
        this float[] actual,
        float[] expected,
        float tolerance = DefaultFloatTolerance,
        string because = "")
    {
        using var _ = new AssertionScope();

        actual.Should().NotBeNull(because);
        expected.Should().NotBeNull(because);
        actual.Length.Should().Be(expected.Length, $"array lengths should match {because}");

        for (int i = 0; i < actual.Length; i++)
        {
            var difference = Math.Abs(actual[i] - expected[i]);
            difference.Should().BeLessThanOrEqualTo(tolerance,
                $"element at index {i} should be within tolerance. Expected: {expected[i]}, Actual: {actual[i]}, Difference: {difference} {because}");
        }
    }

    /// <summary>
    /// Asserts that two double arrays are approximately equal within specified tolerance.
    /// </summary>
    /// <param name="actual">The actual array.</param>
    /// <param name="expected">The expected array.</param>
    /// <param name="tolerance">Maximum allowed difference per element.</param>
    /// <param name="because">Reason for the assertion.</param>
    public static void ShouldBeApproximately(
        this double[] actual,
        double[] expected,
        double tolerance = DefaultDoubleTolerance,
        string because = "")
    {
        using var _ = new AssertionScope();

        actual.Should().NotBeNull(because);
        expected.Should().NotBeNull(because);
        actual.Length.Should().Be(expected.Length, $"array lengths should match {because}");

        for (int i = 0; i < actual.Length; i++)
        {
            var difference = Math.Abs(actual[i] - expected[i]);
            difference.Should().BeLessThanOrEqualTo(tolerance,
                $"element at index {i} should be within tolerance. Expected: {expected[i]}, Actual: {actual[i]}, Difference: {difference} {because}");
        }
    }

    /// <summary>
    /// Asserts that an array contains only finite values (no NaN or Infinity).
    /// </summary>
    /// <param name="array">The array to check.</param>
    /// <param name="because">Reason for the assertion.</param>
    public static void ShouldContainOnlyFiniteValues(this float[] array, string because = "")
    {
        array.Should().NotBeNull(because);

        for (int i = 0; i < array.Length; i++)
        {
            var value = array[i];
            float.IsFinite(value).Should().BeTrue(
                $"element at index {i} should be finite. Value: {value} {because}");
        }
    }

    /// <summary>
    /// Asserts that an array is sorted in ascending order.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="array">The array to check.</param>
    /// <param name="because">Reason for the assertion.</param>
    public static void ShouldBeSorted<T>(this T[] array, string because = "") where T : IComparable<T>
    {
        array.Should().NotBeNull(because);

        for (int i = 1; i < array.Length; i++)
        {
            array[i].CompareTo(array[i - 1]).Should().BeGreaterThanOrEqualTo(0,
                $"element at index {i} should be >= element at index {i - 1}. Values: [{array[i - 1]}] [{array[i]}] {because}");
        }
    }

    /// <summary>
    /// Asserts that two arrays have the same sum within tolerance.
    /// Useful for validating batch operations where order doesn't matter.
    /// </summary>
    /// <param name="actual">The actual array.</param>
    /// <param name="expected">The expected array.</param>
    /// <param name="tolerance">Maximum allowed difference in sum.</param>
    /// <param name="because">Reason for the assertion.</param>
    public static void ShouldHaveSameSum(
        this float[] actual,
        float[] expected,
        float tolerance = DefaultFloatTolerance,
        string because = "")
    {
        actual.Should().NotBeNull(because);
        expected.Should().NotBeNull(because);

        var actualSum = actual.Sum();
        var expectedSum = expected.Sum();
        var difference = Math.Abs(actualSum - expectedSum);

        difference.Should().BeLessThanOrEqualTo(tolerance,
            $"array sums should match within tolerance. Expected: {expectedSum}, Actual: {actualSum}, Difference: {difference} {because}");
    }

    #endregion

    #region Retry Logic

    /// <summary>
    /// Executes an action with retry logic for handling flaky operations.
    /// </summary>
    /// <param name="action">The action to execute.</param>
    /// <param name="maxRetries">Maximum number of retry attempts.</param>
    /// <param name="delayBetweenRetries">Delay between retry attempts.</param>
    /// <param name="onRetry">Optional callback invoked before each retry.</param>
    /// <returns>A task that completes when the action succeeds or max retries is reached.</returns>
    /// <exception cref="AggregateException">Thrown when all retry attempts fail.</exception>
    public static async Task WithRetryAsync(
        Func<Task> action,
        int maxRetries = DefaultRetryCount,
        TimeSpan? delayBetweenRetries = null,
        Action<int, Exception>? onRetry = null)
    {
        var delay = delayBetweenRetries ?? DefaultRetryDelay;
        var exceptions = new List<Exception>();

        for (int attempt = 0; attempt <= maxRetries; attempt++)
        {
            try
            {
                await action().ConfigureAwait(false);
                return;
            }
            catch (Exception ex)
            {
                exceptions.Add(ex);

                if (attempt == maxRetries)
                {
                    throw new AggregateException(
                        $"Operation failed after {maxRetries + 1} attempts.",
                        exceptions);
                }

                onRetry?.Invoke(attempt + 1, ex);
                await Task.Delay(delay).ConfigureAwait(false);
            }
        }
    }

    /// <summary>
    /// Executes a function with retry logic for handling flaky operations.
    /// </summary>
    /// <typeparam name="T">The return type.</typeparam>
    /// <param name="func">The function to execute.</param>
    /// <param name="maxRetries">Maximum number of retry attempts.</param>
    /// <param name="delayBetweenRetries">Delay between retry attempts.</param>
    /// <param name="onRetry">Optional callback invoked before each retry.</param>
    /// <returns>A task that completes with the result when the function succeeds.</returns>
    /// <exception cref="AggregateException">Thrown when all retry attempts fail.</exception>
    public static async Task<T> WithRetryAsync<T>(
        Func<Task<T>> func,
        int maxRetries = DefaultRetryCount,
        TimeSpan? delayBetweenRetries = null,
        Action<int, Exception>? onRetry = null)
    {
        var delay = delayBetweenRetries ?? DefaultRetryDelay;
        var exceptions = new List<Exception>();

        for (int attempt = 0; attempt <= maxRetries; attempt++)
        {
            try
            {
                return await func().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                exceptions.Add(ex);

                if (attempt == maxRetries)
                {
                    throw new AggregateException(
                        $"Operation failed after {maxRetries + 1} attempts.",
                        exceptions);
                }

                onRetry?.Invoke(attempt + 1, ex);
                await Task.Delay(delay).ConfigureAwait(false);
            }
        }

        throw new InvalidOperationException("Unreachable code");
    }

    #endregion

    #region Performance Measurement

    /// <summary>
    /// Measures the execution time of an action.
    /// </summary>
    /// <param name="action">The action to measure.</param>
    /// <returns>A tuple containing the elapsed time and the action result.</returns>
    public static async Task<TimeSpan> MeasureAsync(Func<Task> action)
    {
        var stopwatch = Stopwatch.StartNew();
        await action().ConfigureAwait(false);
        stopwatch.Stop();
        return stopwatch.Elapsed;
    }

    /// <summary>
    /// Measures the execution time of a function.
    /// </summary>
    /// <typeparam name="T">The return type.</typeparam>
    /// <param name="func">The function to measure.</param>
    /// <returns>A tuple containing the result and elapsed time.</returns>
    public static async Task<(T Result, TimeSpan Elapsed)> MeasureAsync<T>(Func<Task<T>> func)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = await func().ConfigureAwait(false);
        stopwatch.Stop();
        return (result, stopwatch.Elapsed);
    }

    /// <summary>
    /// Runs a performance benchmark with multiple iterations.
    /// </summary>
    /// <param name="action">The action to benchmark.</param>
    /// <param name="iterations">Number of iterations.</param>
    /// <param name="warmupIterations">Number of warmup iterations (not included in results).</param>
    /// <returns>Benchmark results including min, max, average, and median times.</returns>
    public static async Task<BenchmarkResult> BenchmarkAsync(
        Func<Task> action,
        int iterations = 100,
        int warmupIterations = 10)
    {
        // Warmup
        for (int i = 0; i < warmupIterations; i++)
        {
            await action().ConfigureAwait(false);
        }

        // Actual benchmark
        var times = new List<TimeSpan>();
        for (int i = 0; i < iterations; i++)
        {
            var elapsed = await MeasureAsync(action).ConfigureAwait(false);
            times.Add(elapsed);
        }

        var sorted = times.OrderBy(t => t).ToList();
        var total = TimeSpan.FromTicks(times.Sum(t => t.Ticks));

        return new BenchmarkResult(
            Iterations: iterations,
            Min: sorted.First(),
            Max: sorted.Last(),
            Average: TimeSpan.FromTicks(total.Ticks / iterations),
            Median: sorted[iterations / 2],
            P95: sorted[(int)(iterations * 0.95)],
            P99: sorted[(int)(iterations * 0.99)]);
    }

    #endregion

    #region Logging and Diagnostics

    /// <summary>
    /// Creates a simple console logger for testing.
    /// </summary>
    /// <param name="categoryName">Logger category name.</param>
    /// <param name="minLevel">Minimum log level.</param>
    /// <returns>A logger instance.</returns>
    public static ILogger CreateTestLogger(string categoryName, LogLevel minLevel = LogLevel.Information)
    {
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.SetMinimumLevel(minLevel);
            builder.AddConsole();
        });

        return loggerFactory.CreateLogger(categoryName);
    }

    /// <summary>
    /// Captures and returns console output during action execution.
    /// </summary>
    /// <param name="action">The action to execute.</param>
    /// <returns>The captured console output.</returns>
    public static async Task<string> CaptureConsoleOutputAsync(Func<Task> action)
    {
        var originalOut = Console.Out;
        try
        {
            using var stringWriter = new System.IO.StringWriter();
            Console.SetOut(stringWriter);
            await action().ConfigureAwait(false);
            return stringWriter.ToString();
        }
        finally
        {
            Console.SetOut(originalOut);
        }
    }

    /// <summary>
    /// Formats byte size in human-readable format.
    /// </summary>
    /// <param name="bytes">Size in bytes.</param>
    /// <returns>Formatted string (e.g., "1.5 MB").</returns>
    public static string FormatBytes(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB", "TB" };
        double len = bytes;
        int order = 0;
        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len /= 1024;
        }
        return $"{len:0.##} {sizes[order]}";
    }

    /// <summary>
    /// Formats duration in human-readable format.
    /// </summary>
    /// <param name="duration">Duration to format.</param>
    /// <returns>Formatted string (e.g., "123.45 ms").</returns>
    public static string FormatDuration(TimeSpan duration)
    {
        if (duration.TotalSeconds >= 1)
            return $"{duration.TotalSeconds:0.##} s";
        if (duration.TotalMilliseconds >= 1)
            return $"{duration.TotalMilliseconds:0.##} ms";
        return $"{duration.TotalMicroseconds:0.##} μs";
    }

    #endregion

    #region Cancellation Helpers

    /// <summary>
    /// Creates a cancellation token that cancels after specified timeout.
    /// </summary>
    /// <param name="timeout">Timeout duration.</param>
    /// <returns>A cancellation token.</returns>
    public static CancellationToken CreateTimeoutToken(TimeSpan timeout)
    {
        var cts = new CancellationTokenSource(timeout);
        return cts.Token;
    }

    /// <summary>
    /// Executes an action with timeout protection.
    /// </summary>
    /// <param name="action">The action to execute.</param>
    /// <param name="timeout">Maximum execution time.</param>
    /// <returns>A task that completes when the action finishes or times out.</returns>
    /// <exception cref="TimeoutException">Thrown when the operation times out.</exception>
    public static async Task WithTimeoutAsync(Func<CancellationToken, Task> action, TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        try
        {
            await action(cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cts.IsCancellationRequested)
        {
            throw new TimeoutException($"Operation timed out after {timeout.TotalSeconds} seconds.");
        }
    }

    #endregion
}

/// <summary>
/// Results from a performance benchmark.
/// </summary>
/// <param name="Iterations">Number of iterations executed.</param>
/// <param name="Min">Minimum execution time.</param>
/// <param name="Max">Maximum execution time.</param>
/// <param name="Average">Average execution time.</param>
/// <param name="Median">Median execution time.</param>
/// <param name="P95">95th percentile execution time.</param>
/// <param name="P99">99th percentile execution time.</param>
public sealed record BenchmarkResult(
    int Iterations,
    TimeSpan Min,
    TimeSpan Max,
    TimeSpan Average,
    TimeSpan Median,
    TimeSpan P95,
    TimeSpan P99)
{
    /// <summary>
    /// Returns a formatted string representation of the benchmark results.
    /// </summary>
    public override string ToString()
    {
        return $"Benchmark Results ({Iterations} iterations):\n" +
               $"  Min:     {GpuTestHelpers.FormatDuration(Min)}\n" +
               $"  Max:     {GpuTestHelpers.FormatDuration(Max)}\n" +
               $"  Average: {GpuTestHelpers.FormatDuration(Average)}\n" +
               $"  Median:  {GpuTestHelpers.FormatDuration(Median)}\n" +
               $"  P95:     {GpuTestHelpers.FormatDuration(P95)}\n" +
               $"  P99:     {GpuTestHelpers.FormatDuration(P99)}";
    }
}
