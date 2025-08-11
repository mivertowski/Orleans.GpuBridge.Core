using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

namespace Orleans.GpuBridge.Diagnostics;

public static class TelemetryExtensions
{
    public static IServiceCollection AddGpuTelemetry(
        this IServiceCollection services,
        Action<GpuTelemetryOptions>? configure = null)
    {
        var options = new GpuTelemetryOptions();
        configure?.Invoke(options);
        
        // Register core telemetry services
        services.AddSingleton<IGpuTelemetry, GpuTelemetry>();
        services.AddSingleton<IGpuMetricsCollector, GpuMetricsCollector>();
        services.AddHostedService<GpuMetricsCollector>();
        
        // Configure metrics collection options
        services.Configure<GpuMetricsOptions>(opt =>
        {
            opt.CollectionInterval = options.MetricsCollectionInterval;
            opt.EnableSystemMetrics = options.EnableSystemMetrics;
            opt.EnableGpuMetrics = options.EnableGpuMetrics;
            opt.EnableDetailedLogging = options.EnableDetailedLogging;
            opt.MaxDevices = options.MaxDevices;
        });
        
        // Configure OpenTelemetry
        services.AddOpenTelemetry()
            .ConfigureResource(resource =>
            {
                resource
                    .AddService(
                        serviceName: options.ServiceName,
                        serviceVersion: options.ServiceVersion,
                        serviceInstanceId: options.ServiceInstanceId ?? Environment.MachineName)
                    .AddAttributes(new Dictionary<string, object>
                    {
                        ["deployment.environment"] = options.Environment,
                        ["gpu.enabled"] = true,
                        ["gpu.bridge.version"] = typeof(TelemetryExtensions).Assembly.GetName().Version?.ToString() ?? "unknown"
                    });
            })
            .WithMetrics(metrics =>
            {
                metrics
                    .AddMeter("Orleans.GpuBridge")
                    .AddRuntimeInstrumentation()
                    .AddProcessInstrumentation()
                    .AddAspNetCoreInstrumentation();
                
                // Add exporters based on configuration
                if (options.EnablePrometheusExporter)
                {
                    metrics.AddPrometheusExporter();
                }
                
                if (!string.IsNullOrEmpty(options.OtlpEndpoint))
                {
                    metrics.AddOtlpExporter(otlp =>
                    {
                        otlp.Endpoint = new Uri(options.OtlpEndpoint);
                        otlp.Protocol = options.OtlpProtocol;
                    });
                }
                
                if (options.EnableConsoleExporter)
                {
                    metrics.AddConsoleExporter();
                }
                
                // Configure views for histograms
                metrics.AddView(
                    instrumentName: "gpu.kernel.latency",
                    new ExplicitBucketHistogramConfiguration
                    {
                        Boundaries = new double[] { 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000 }
                    });
                
                metrics.AddView(
                    instrumentName: "gpu.transfer.throughput",
                    new ExplicitBucketHistogramConfiguration
                    {
                        Boundaries = new double[] { 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100 }
                    });
            })
            .WithTracing(tracing =>
            {
                tracing
                    .AddSource("Orleans.GpuBridge")
                    .AddAspNetCoreInstrumentation()
                    .AddHttpClientInstrumentation()
                    .SetSampler(new TraceIdRatioBasedSampler(options.TracingSamplingRatio));
                
                // Add exporters
                if (!string.IsNullOrEmpty(options.OtlpEndpoint))
                {
                    tracing.AddOtlpExporter(otlp =>
                    {
                        otlp.Endpoint = new Uri(options.OtlpEndpoint);
                        otlp.Protocol = options.OtlpProtocol;
                    });
                }
                
                if (!string.IsNullOrEmpty(options.JaegerEndpoint))
                {
                    tracing.AddJaegerExporter(jaeger =>
                    {
                        jaeger.AgentHost = new Uri(options.JaegerEndpoint).Host;
                        jaeger.AgentPort = new Uri(options.JaegerEndpoint).Port;
                    });
                }
                
                if (options.EnableConsoleExporter)
                {
                    tracing.AddConsoleExporter();
                }
            });
        
        return services;
    }
    
    public static IHostBuilder UseGpuTelemetry(
        this IHostBuilder hostBuilder,
        Action<GpuTelemetryOptions>? configure = null)
    {
        return hostBuilder.ConfigureServices((context, services) =>
        {
            services.AddGpuTelemetry(configure);
        });
    }
}

public class GpuTelemetryOptions
{
    // Service identification
    public string ServiceName { get; set; } = "Orleans.GpuBridge";
    public string ServiceVersion { get; set; } = "1.0.0";
    public string? ServiceInstanceId { get; set; }
    public string Environment { get; set; } = "production";
    
    // Metrics configuration
    public bool EnableGpuMetrics { get; set; } = true;
    public bool EnableSystemMetrics { get; set; } = true;
    public TimeSpan MetricsCollectionInterval { get; set; } = TimeSpan.FromSeconds(10);
    public int MaxDevices { get; set; } = 8;
    public bool EnableDetailedLogging { get; set; } = false;
    
    // Tracing configuration
    public double TracingSamplingRatio { get; set; } = 0.1;
    
    // Exporters
    public bool EnablePrometheusExporter { get; set; } = true;
    public bool EnableConsoleExporter { get; set; } = false;
    public string? OtlpEndpoint { get; set; }
    public OpenTelemetry.Exporter.OtlpExportProtocol OtlpProtocol { get; set; } = OpenTelemetry.Exporter.OtlpExportProtocol.Grpc;
    public string? JaegerEndpoint { get; set; }
}