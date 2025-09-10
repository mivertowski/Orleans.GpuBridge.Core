using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Configuration;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Resilience;
using Orleans.GpuBridge.Resilience.Policies;
using Orleans.GpuBridge.Runtime;

namespace Orleans.GpuBridge.Examples.Resilience;

/// <summary>
/// Comprehensive example demonstrating GPU bridge resilience patterns
/// </summary>
class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("🛡️  Orleans GPU Bridge - Resilience Patterns Example");
        Console.WriteLine("=====================================================");

        var host = CreateHostBuilder(args).Build();
        
        try
        {
            await host.StartAsync();
            
            var example = host.Services.GetRequiredService<ResilienceExample>();
            await example.RunAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Application failed: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
        finally
        {
            await host.StopAsync();
        }

        Console.WriteLine("\n✅ Resilience example completed. Press any key to exit...");
        Console.ReadKey();
    }

    static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureAppConfiguration((context, config) =>
            {
                config.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);
                config.AddEnvironmentVariables();
                config.AddCommandLine(args);
            })
            .ConfigureServices((context, services) =>
            {
                // Configure logging
                services.AddLogging(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(LogLevel.Information);
                });

                // Add GPU Bridge with comprehensive resilience
                services.AddGpuBridge(options =>
                {
                    options.PreferGpu = true;
                    options.EnableMetrics = true;
                })
                .AddGpuBridgeResilience(context.Configuration, options =>
                {
                    // Configure retry policy
                    options.RetryOptions.MaxAttempts = 5;
                    options.RetryOptions.BaseDelay = TimeSpan.FromMilliseconds(250);
                    options.RetryOptions.MaxDelay = TimeSpan.FromSeconds(10);
                    options.RetryOptions.UseExponentialBackoff = true;
                    options.RetryOptions.UseJitter = true;

                    // Configure circuit breaker
                    options.CircuitBreakerOptions.FailureRatio = 0.6;
                    options.CircuitBreakerOptions.MinimumThroughput = 5;
                    options.CircuitBreakerOptions.BreakDuration = TimeSpan.FromSeconds(30);

                    // Configure bulkhead isolation
                    options.BulkheadOptions.MaxConcurrentOperations = 8;
                    options.BulkheadOptions.MaxQueuedOperations = 20;

                    // Configure timeouts
                    options.TimeoutOptions.KernelExecution = TimeSpan.FromMinutes(2);
                    options.TimeoutOptions.DeviceOperation = TimeSpan.FromSeconds(15);
                })
                .AddGpuRateLimit(options =>
                {
                    options.MaxRequests = 50;
                    options.TimeWindow = TimeSpan.FromMinutes(1);
                    options.Algorithm = RateLimitingAlgorithm.TokenBucket;
                    options.TokenRefillRate = 5.0;
                    options.MaxBurstSize = 10;
                })
                .AddGpuChaosEngineering(options =>
                {
                    options.Enabled = true; // Enable for demonstration
                    options.FaultInjectionProbability = 0.1; // 10% fault injection
                    
                    options.LatencyInjection.Enabled = true;
                    options.LatencyInjection.InjectionProbability = 0.2;
                    options.LatencyInjection.MinLatency = TimeSpan.FromMilliseconds(100);
                    options.LatencyInjection.MaxLatency = TimeSpan.FromSeconds(2);
                    
                    options.ExceptionInjection.Enabled = true;
                    options.ExceptionInjection.InjectionProbability = 0.05;
                })
                .AddGpuTelemetry(options =>
                {
                    options.RetentionPeriod = TimeSpan.FromHours(2);
                    options.HealthCheckInterval = TimeSpan.FromSeconds(10);
                    options.EnableDetailedTracing = true;
                });

                // Add Orleans client (for this example)
                services.AddOrleansClient(clientBuilder =>
                {
                    clientBuilder.UseLocalhostClustering();
                    clientBuilder.ConfigureLogging(logging => logging.AddConsole());
                });

                // Add example service
                services.AddSingleton<ResilienceExample>();
            });
}

/// <summary>
/// Main resilience demonstration class
/// </summary>
public sealed class ResilienceExample
{
    private readonly ILogger<ResilienceExample> _logger;
    private readonly IGpuBridge _gpuBridge;
    private readonly GpuResiliencePolicy _resiliencePolicy;
    private readonly IResilienceHealthMonitor _healthMonitor;
    private readonly IClusterClient _clusterClient;

    public ResilienceExample(
        ILogger<ResilienceExample> logger,
        IGpuBridge gpuBridge,
        GpuResiliencePolicy resiliencePolicy,
        IResilienceHealthMonitor healthMonitor,
        IClusterClient clusterClient)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _gpuBridge = gpuBridge ?? throw new ArgumentNullException(nameof(gpuBridge));
        _resiliencePolicy = resiliencePolicy ?? throw new ArgumentNullException(nameof(resiliencePolicy));
        _healthMonitor = healthMonitor ?? throw new ArgumentNullException(nameof(healthMonitor));
        _clusterClient = clusterClient ?? throw new ArgumentNullException(nameof(clusterClient));
    }

    public async Task RunAsync()
    {
        Console.WriteLine("\n🔄 Starting resilience pattern demonstrations...\n");

        // 1. Basic resilience demonstration
        await DemonstrateBasicResilienceAsync();

        // 2. Circuit breaker demonstration
        await DemonstrateCircuitBreakerAsync();

        // 3. Rate limiting demonstration
        await DemonstrateRateLimitingAsync();

        // 4. Fallback chain demonstration
        await DemonstrateFallbackChainAsync();

        // 5. Chaos engineering demonstration
        await DemonstrateChaosEngineeringAsync();

        // 6. Health monitoring demonstration
        await DemonstrateHealthMonitoringAsync();

        // 7. Telemetry and metrics demonstration
        await DemonstrateTelemetryAsync();

        Console.WriteLine("\n🎉 All resilience demonstrations completed successfully!");
    }

    private async Task DemonstrateBasicResilienceAsync()
    {
        Console.WriteLine("📋 1. Basic Resilience Patterns");
        Console.WriteLine("   - Automatic retries with exponential backoff");
        Console.WriteLine("   - Timeout handling");
        Console.WriteLine("   - Error logging and context");

        try
        {
            // Get a kernel that might fail occasionally
            var kernel = await _gpuBridge.GetKernelAsync<float[], float>(new("demo-vector-add"));
            
            // Submit work that will be retried automatically if it fails
            var testData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            var handle = await kernel.SubmitBatchAsync(new[] { testData });
            
            var results = new List<float>();
            await foreach (var result in kernel.ReadResultsAsync(handle))
            {
                results.Add(result);
            }

            Console.WriteLine($"   ✅ Successfully processed {results.Count} results with automatic retry handling");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"   ❌ Operation failed after all retries: {ex.Message}");
        }

        await Task.Delay(1000); // Brief pause for demonstration
        Console.WriteLine();
    }

    private async Task DemonstrateCircuitBreakerAsync()
    {
        Console.WriteLine("⚡ 2. Circuit Breaker Pattern");
        Console.WriteLine("   - Automatic failure detection");
        Console.WriteLine("   - Circuit opening to prevent cascading failures");
        Console.WriteLine("   - Automatic recovery attempts");

        try
        {
            // Simulate multiple operations that might trigger circuit breaker
            for (int i = 0; i < 10; i++)
            {
                try
                {
                    await _resiliencePolicy.ExecuteKernelOperationAsync(
                        async (ct) =>
                        {
                            // Simulate operation that might fail
                            if (i % 3 == 0) // Fail every 3rd operation
                            {
                                throw new InvalidOperationException($"Simulated failure {i}");
                            }
                            
                            await Task.Delay(100, ct);
                            return $"Operation {i} succeeded";
                        },
                        $"CircuitBreakerDemo_Op{i}");

                    Console.WriteLine($"   ✅ Operation {i} succeeded");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️  Operation {i} failed or circuit breaker opened: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"   ❌ Circuit breaker demonstration failed: {ex.Message}");
        }

        await Task.Delay(1000);
        Console.WriteLine();
    }

    private async Task DemonstrateRateLimitingAsync()
    {
        Console.WriteLine("🚦 3. Rate Limiting");
        Console.WriteLine("   - Token bucket algorithm");
        Console.WriteLine("   - Backpressure handling");
        Console.WriteLine("   - Graceful degradation under load");

        try
        {
            // Simulate burst of requests that will trigger rate limiting
            var tasks = new List<Task>();
            
            for (int i = 0; i < 15; i++) // More requests than rate limit allows
            {
                tasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        await _resiliencePolicy.ExecuteKernelOperationAsync(
                            async (ct) =>
                            {
                                await Task.Delay(50, ct); // Simulate work
                                return $"Request processed";
                            },
                            $"RateLimitDemo_Req{i}");
                        
                        Console.WriteLine($"   ✅ Request {i} processed");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   🚦 Request {i} rate limited: {ex.Message}");
                    }
                }));
            }

            await Task.WhenAll(tasks);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"   ❌ Rate limiting demonstration failed: {ex.Message}");
        }

        await Task.Delay(1000);
        Console.WriteLine();
    }

    private async Task DemonstrateFallbackChainAsync()
    {
        Console.WriteLine("🔄 4. Fallback Chain");
        Console.WriteLine("   - GPU -> CPU -> Error degradation");
        Console.WriteLine("   - Automatic recovery");
        Console.WriteLine("   - Performance monitoring");

        try
        {
            // Simulate operations that demonstrate fallback chain
            var kernel = await _gpuBridge.GetKernelAsync<float[], float>(new("fallback-demo"));
            
            for (int i = 0; i < 5; i++)
            {
                try
                {
                    var testData = new float[] { 1.0f * i, 2.0f * i, 3.0f * i };
                    var handle = await kernel.SubmitBatchAsync(new[] { testData });
                    
                    await foreach (var result in kernel.ReadResultsAsync(handle))
                    {
                        Console.WriteLine($"   ✅ Fallback chain operation {i} result: {result}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ❌ Fallback chain exhausted for operation {i}: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"   ❌ Fallback chain demonstration failed: {ex.Message}");
        }

        await Task.Delay(1000);
        Console.WriteLine();
    }

    private async Task DemonstrateChaosEngineeringAsync()
    {
        Console.WriteLine("🎭 5. Chaos Engineering");
        Console.WriteLine("   - Random fault injection");
        Console.WriteLine("   - Latency injection");
        Console.WriteLine("   - Resource exhaustion simulation");

        try
        {
            // Run operations with chaos engineering enabled
            for (int i = 0; i < 8; i++)
            {
                try
                {
                    var result = await _resiliencePolicy.ExecuteKernelOperationAsync(
                        async (ct) =>
                        {
                            // This operation might have chaos faults injected
                            await Task.Delay(100, ct);
                            return $"Chaos test operation {i} completed";
                        },
                        $"ChaosDemo_Op{i}");

                    Console.WriteLine($"   ✅ {result}");
                }
                catch (Exception ex)
                {
                    if (ex.Message.Contains("[CHAOS]"))
                    {
                        Console.WriteLine($"   🎭 Chaos fault injected in operation {i}: {ex.Message}");
                    }
                    else
                    {
                        Console.WriteLine($"   ❌ Operation {i} failed: {ex.Message}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"   ❌ Chaos engineering demonstration failed: {ex.Message}");
        }

        await Task.Delay(1000);
        Console.WriteLine();
    }

    private async Task DemonstrateHealthMonitoringAsync()
    {
        Console.WriteLine("🏥 6. Health Monitoring");
        Console.WriteLine("   - Component health tracking");
        Console.WriteLine("   - Real-time status reporting");
        Console.WriteLine("   - Degradation detection");

        try
        {
            var healthStatus = await _healthMonitor.GetHealthStatusAsync();
            var healthReport = await _healthMonitor.GetHealthReportAsync();

            Console.WriteLine($"   📊 Overall health status: {healthStatus}");
            Console.WriteLine($"   📈 Component count: {healthReport.ComponentHealth.Count}");
            Console.WriteLine($"   🔧 Bulkhead utilization: {healthReport.BulkheadUtilization:F1}%");
            Console.WriteLine($"   ⚡ Circuit breakers: {healthReport.CircuitBreakerStates.Count}");
            Console.WriteLine($"   🔄 Fallback levels: {healthReport.FallbackLevels.Count}");
            Console.WriteLine($"   📅 Last updated: {healthReport.Timestamp:HH:mm:ss}");

            // Display recent health events
            if (healthReport.RecentEvents.Any())
            {
                Console.WriteLine("   📋 Recent health events:");
                foreach (var evt in healthReport.RecentEvents.Take(3))
                {
                    Console.WriteLine($"      - {evt.Timestamp:HH:mm:ss}: {evt.Component} {evt.PreviousHealth} -> {evt.NewHealth}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"   ❌ Health monitoring demonstration failed: {ex.Message}");
        }

        await Task.Delay(1000);
        Console.WriteLine();
    }

    private async Task DemonstrateTelemetryAsync()
    {
        Console.WriteLine("📊 7. Telemetry and Metrics");
        Console.WriteLine("   - Operation performance tracking");
        Console.WriteLine("   - Failure rate monitoring");
        Console.WriteLine("   - Resource utilization metrics");

        try
        {
            // Get bulkhead metrics
            var bulkheadMetrics = _resiliencePolicy.GetBulkheadMetrics();
            
            Console.WriteLine($"   🎯 Bulkhead slots: {bulkheadMetrics.InUseSlots}/{bulkheadMetrics.TotalSlots} in use");
            Console.WriteLine($"   📈 Utilization: {bulkheadMetrics.UtilizationPercentage:F1}%");
            Console.WriteLine($"   🆓 Available slots: {bulkheadMetrics.AvailableSlots}");

            // Demonstrate metric collection
            for (int i = 0; i < 3; i++)
            {
                var startTime = DateTimeOffset.UtcNow;
                try
                {
                    await Task.Delay(Random.Shared.Next(50, 200));
                    var duration = DateTimeOffset.UtcNow - startTime;
                    Console.WriteLine($"   ⏱️  Metric collection {i}: {duration.TotalMilliseconds:F0}ms");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ❌ Metric collection {i} failed: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"   ❌ Telemetry demonstration failed: {ex.Message}");
        }

        await Task.Delay(1000);
        Console.WriteLine();
    }
}