// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Resilience.RateLimit;

namespace Orleans.GpuBridge.Resilience.Tests.RateLimit;

/// <summary>
/// Tests for <see cref="TokenBucketRateLimiter"/> class.
/// </summary>
public class TokenBucketRateLimiterTests : IDisposable
{
    private readonly Mock<ILogger<TokenBucketRateLimiter>> _loggerMock;
    private readonly RateLimitingOptions _options;
    private readonly TokenBucketRateLimiter _rateLimiter;

    public TokenBucketRateLimiterTests()
    {
        _loggerMock = new Mock<ILogger<TokenBucketRateLimiter>>();
        _options = new RateLimitingOptions
        {
            Enabled = true,
            TokenRefillRate = 10.0, // 10 tokens per second
            MaxBurstSize = 5
        };

        _rateLimiter = new TokenBucketRateLimiter(
            _loggerMock.Object,
            Options.Create(_options));
    }

    [Fact]
    public void Constructor_ShouldInitializeSuccessfully()
    {
        // Assert
        _rateLimiter.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_ShouldThrowOnNullLogger()
    {
        // Act
        var act = () => new TokenBucketRateLimiter(null!, Options.Create(_options));

        // Assert
        act.Should().Throw<ArgumentNullException>().WithParameterName("logger");
    }

    [Fact]
    public void Constructor_ShouldThrowOnNullOptions()
    {
        // Act
        var act = () => new TokenBucketRateLimiter(_loggerMock.Object, null!);

        // Assert
        act.Should().Throw<ArgumentNullException>().WithParameterName("options");
    }

    [Fact]
    public void Constructor_ShouldThrowOnInvalidRefillRate()
    {
        // Arrange
        var invalidOptions = new RateLimitingOptions
        {
            TokenRefillRate = 0
        };

        // Act
        var act = () => new TokenBucketRateLimiter(
            _loggerMock.Object,
            Options.Create(invalidOptions));

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Constructor_ShouldThrowOnInvalidBurstSize()
    {
        // Arrange
        var invalidOptions = new RateLimitingOptions
        {
            TokenRefillRate = 10,
            MaxBurstSize = 0
        };

        // Act
        var act = () => new TokenBucketRateLimiter(
            _loggerMock.Object,
            Options.Create(invalidOptions));

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public async Task TryAcquireAsync_ShouldSucceedWithAvailableTokens()
    {
        // Act
        var result = await _rateLimiter.TryAcquireAsync();

        // Assert
        result.Should().BeTrue();
    }

    [Fact]
    public async Task TryAcquireAsync_ShouldFailWhenExhausted()
    {
        // Arrange - exhaust all tokens
        for (int i = 0; i < _options.MaxBurstSize; i++)
        {
            await _rateLimiter.TryAcquireAsync();
        }

        // Act
        var result = await _rateLimiter.TryAcquireAsync();

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public async Task TryAcquireAsync_ShouldReturnTrueWhenDisabled()
    {
        // Arrange
        var disabledOptions = new RateLimitingOptions
        {
            Enabled = false,
            TokenRefillRate = 10,
            MaxBurstSize = 1
        };
        using var rateLimiter = new TokenBucketRateLimiter(
            _loggerMock.Object,
            Options.Create(disabledOptions));

        // Act - try to acquire more than available
        var result1 = await rateLimiter.TryAcquireAsync();
        var result2 = await rateLimiter.TryAcquireAsync();
        var result3 = await rateLimiter.TryAcquireAsync();

        // Assert - all should succeed when disabled
        result1.Should().BeTrue();
        result2.Should().BeTrue();
        result3.Should().BeTrue();
    }

    [Fact]
    public async Task TryAcquireAsync_ShouldAcquireMultipleTokens()
    {
        // Act
        var result = await _rateLimiter.TryAcquireAsync(3);

        // Assert
        result.Should().BeTrue();

        // Verify remaining tokens
        var metrics = _rateLimiter.GetMetrics();
        metrics.AvailableTokens.Should().BeLessThan(_options.MaxBurstSize);
    }

    [Fact]
    public async Task ExecuteAsync_ShouldExecuteOperation()
    {
        // Arrange
        var expectedResult = 42;

        // Act
        var result = await _rateLimiter.ExecuteAsync(
            async ct =>
            {
                await Task.Delay(1, ct);
                return expectedResult;
            });

        // Assert
        result.Should().Be(expectedResult);
    }

    [Fact]
    public void GetMetrics_ShouldReturnValidMetrics()
    {
        // Act
        var metrics = _rateLimiter.GetMetrics();

        // Assert
        metrics.TotalRequests.Should().Be(0);
        metrics.RejectedRequests.Should().Be(0);
        metrics.RejectionRate.Should().Be(0);
        metrics.AvailableTokens.Should().BeGreaterThan(0);
        metrics.MaxTokens.Should().Be(_options.MaxBurstSize);
        metrics.RefillRate.Should().Be(_options.TokenRefillRate);
    }

    [Fact]
    public async Task GetMetrics_ShouldTrackRequests()
    {
        // Arrange
        await _rateLimiter.TryAcquireAsync();
        await _rateLimiter.TryAcquireAsync();

        // Act
        var metrics = _rateLimiter.GetMetrics();

        // Assert
        metrics.TotalRequests.Should().Be(2);
    }

    [Fact]
    public async Task GetMetrics_ShouldTrackRejections()
    {
        // Arrange - exhaust tokens
        for (int i = 0; i < _options.MaxBurstSize; i++)
        {
            await _rateLimiter.TryAcquireAsync();
        }
        // Try to acquire when exhausted
        await _rateLimiter.TryAcquireAsync();

        // Act
        var metrics = _rateLimiter.GetMetrics();

        // Assert
        metrics.RejectedRequests.Should().Be(1);
        metrics.RejectionRate.Should().BeGreaterThan(0);
    }

    [Fact]
    public void ResetStats_ShouldClearCounters()
    {
        // Arrange
        _rateLimiter.TryAcquireAsync().Wait();

        // Act
        _rateLimiter.ResetStats();
        var metrics = _rateLimiter.GetMetrics();

        // Assert
        metrics.TotalRequests.Should().Be(0);
        metrics.RejectedRequests.Should().Be(0);
    }

    [Fact]
    public void AddTokens_ShouldIncreaseAvailableTokens()
    {
        // Arrange - exhaust some tokens
        _rateLimiter.TryAcquireAsync(3).Wait();
        var beforeMetrics = _rateLimiter.GetMetrics();

        // Act
        _rateLimiter.AddTokens(2);
        var afterMetrics = _rateLimiter.GetMetrics();

        // Assert
        afterMetrics.AvailableTokens.Should().BeGreaterThan(beforeMetrics.AvailableTokens);
    }

    [Fact]
    public void AddTokens_ShouldNotExceedMaxBurstSize()
    {
        // Act
        _rateLimiter.AddTokens(100);
        var metrics = _rateLimiter.GetMetrics();

        // Assert
        metrics.AvailableTokens.Should().BeLessThanOrEqualTo(_options.MaxBurstSize);
    }

    [Fact]
    public void Dispose_ShouldNotThrow()
    {
        // Act
        var act = () => _rateLimiter.Dispose();

        // Assert
        act.Should().NotThrow();
    }

    public void Dispose()
    {
        _rateLimiter?.Dispose();
    }
}
