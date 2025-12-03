using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using Orleans.GpuBridge.Abstractions.Metrics;
using Orleans.GpuBridge.Diagnostics.Configuration;
using Orleans.GpuBridge.Diagnostics.Implementation;
using Orleans.GpuBridge.Diagnostics.Interfaces;

namespace Orleans.GpuBridge.Diagnostics;

/// <summary>
/// Extension methods for configuring GPU telemetry and observability in dependency injection containers.
/// These extensions set up comprehensive monitoring including OpenTelemetry metrics, distributed tracing,
/// and GPU-specific performance monitoring for Orleans GPU Bridge applications.
/// </summary>
public static class TelemetryExtensions
{
    /// <summary>
    /// Adds GPU telemetry services and OpenTelemetry configuration to the service collection.
    /// This method registers all necessary services for GPU performance monitoring, metrics collection,
    /// and distributed tracing with industry-standard observability tools.
    /// </summary>
    /// <param name="services">The service collection to configure.</param>
    /// <param name="configure">Optional action to configure telemetry options and exporters.</param>
    /// <returns>The configured service collection for method chaining.</returns>
    /// <remarks>
    /// This method configures:
    /// - GPU-specific telemetry services (metrics collection, performance monitoring)
    /// - OpenTelemetry metrics with custom histogram buckets for GPU workloads
    /// - Distributed tracing with GPU kernel execution spans
    /// - Multiple export targets (Prometheus, OTLP, Jaeger, Console)
    /// - Runtime and process instrumentation
    /// 
    /// The telemetry system provides comprehensive monitoring of:
    /// - Kernel execution performance and success rates
    /// - GPU memory usage and transfer rates
    /// - Device utilization and thermal characteristics
    /// - Pipeline stage execution and bottlenecks
    /// - Orleans grain activation and lifecycle events
    /// </remarks>
    /// <example>
    /// <code>
    /// services.AddGpuTelemetry(options =>
    /// {
    ///     options.ServiceName = "MyGpuApp";
    ///     options.EnablePrometheusExporter = true;
    ///     options.OtlpEndpoint = "http://jaeger:14268/api/traces";
    ///     options.TracingSamplingRatio = 0.1; // 10% sampling
    /// });
    /// </code>
    /// </example>
    public static IServiceCollection AddGpuTelemetry(
        this IServiceCollection services,
        Action<GpuTelemetryOptions>? configure = null)
    {
        var options = new GpuTelemetryOptions();
        configure?.Invoke(options);

        // Register core telemetry services
        services.AddSingleton<IGpuTelemetry, GpuTelemetry>();
        services.AddSingleton<IGpuMemoryTelemetryProvider, GpuMemoryTelemetryProvider>();
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
                    .AddMeter("Orleans.GpuBridge.GrainMemory")
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

    /// <summary>
    /// Configures GPU telemetry for the host builder, integrating with the application's service configuration.
    /// This extension method provides a convenient way to add GPU observability to hosted applications
    /// such as ASP.NET Core applications, worker services, or console applications.
    /// </summary>
    /// <param name="hostBuilder">The host builder to configure.</param>
    /// <param name="configure">Optional action to configure telemetry options and exporters.</param>
    /// <returns>The configured host builder for method chaining.</returns>
    /// <remarks>
    /// This method is a convenience wrapper that calls <see cref="AddGpuTelemetry"/> during service configuration.
    /// It's particularly useful for applications using the Generic Host pattern where services are configured
    /// as part of the host building process.
    /// </remarks>
    /// <example>
    /// <code>
    /// Host.CreateDefaultBuilder(args)
    ///     .UseGpuTelemetry(options =>
    ///     {
    ///         options.ServiceName = "GpuWorkerService";
    ///         options.EnablePrometheusExporter = true;
    ///     })
    ///     .Build()
    ///     .Run();
    /// </code>
    /// </example>
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

/// <summary>
/// Configuration options for GPU telemetry services, OpenTelemetry exporters, and monitoring behavior.
/// This class provides comprehensive control over all aspects of GPU observability including
/// service identification, metrics collection, tracing, and export targets.
/// </summary>
public class GpuTelemetryOptions
{
    #region Service Identification

    /// <summary>
    /// Gets or sets the service name used for OpenTelemetry resource identification.
    /// This name appears in telemetry data and dashboards to identify the application.
    /// </summary>
    /// <value>The service name. Default is "Orleans.GpuBridge".</value>
    public string ServiceName { get; set; } = "Orleans.GpuBridge";

    /// <summary>
    /// Gets or sets the service version for telemetry resource attributes.
    /// This helps track performance changes across application versions.
    /// </summary>
    /// <value>The service version. Default is "1.0.0".</value>
    public string ServiceVersion { get; set; } = "1.0.0";

    /// <summary>
    /// Gets or sets the unique service instance identifier.
    /// When not specified, the machine name is used as the instance identifier.
    /// </summary>
    /// <value>The service instance ID, or <c>null</c> to use machine name.</value>
    public string? ServiceInstanceId { get; set; }

    /// <summary>
    /// Gets or sets the deployment environment identifier (e.g., "development", "staging", "production").
    /// This attribute helps filter and organize telemetry data by environment.
    /// </summary>
    /// <value>The environment name. Default is "production".</value>
    public string Environment { get; set; } = "production";

    #endregion

    #region Metrics Configuration

    /// <summary>
    /// Gets or sets a value indicating whether GPU-specific metrics collection is enabled.
    /// When <c>true</c>, collects detailed GPU performance metrics, utilization, and hardware statistics.
    /// </summary>
    /// <value><c>true</c> to enable GPU metrics; otherwise, <c>false</c>. Default is <c>true</c>.</value>
    public bool EnableGpuMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether system-level metrics collection is enabled.
    /// When <c>true</c>, collects runtime, process, and system performance metrics.
    /// </summary>
    /// <value><c>true</c> to enable system metrics; otherwise, <c>false</c>. Default is <c>true</c>.</value>
    public bool EnableSystemMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets the interval for collecting GPU hardware metrics from devices.
    /// This controls how frequently GPU utilization, temperature, and memory usage are sampled.
    /// </summary>
    /// <value>The collection interval. Default is 10 seconds.</value>
    /// <remarks>
    /// Shorter intervals provide more granular monitoring but increase system overhead.
    /// Typical values: 5-30 seconds depending on monitoring requirements.
    /// </remarks>
    public TimeSpan MetricsCollectionInterval { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Gets or sets the maximum number of GPU devices to monitor.
    /// This limits resource usage in systems with many GPUs.
    /// </summary>
    /// <value>The maximum device count. Default is 8.</value>
    public int MaxDevices { get; set; } = 8;

    /// <summary>
    /// Gets or sets a value indicating whether detailed diagnostic logging is enabled.
    /// When <c>true</c>, provides verbose logging of telemetry operations for troubleshooting.
    /// </summary>
    /// <value><c>true</c> to enable detailed logging; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    public bool EnableDetailedLogging { get; set; } = false;

    #endregion

    #region Tracing Configuration

    /// <summary>
    /// Gets or sets the sampling ratio for distributed tracing (0.0 to 1.0).
    /// This controls what percentage of operations are traced to manage overhead and storage costs.
    /// </summary>
    /// <value>The sampling ratio. Default is 0.1 (10% sampling).</value>
    /// <remarks>
    /// Values: 0.0 = no tracing, 1.0 = trace everything, 0.1 = trace 10% of operations.
    /// Higher sampling provides more detailed tracing but increases overhead and storage requirements.
    /// </remarks>
    public double TracingSamplingRatio { get; set; } = 0.1;

    #endregion

    #region Exporters Configuration

    /// <summary>
    /// Gets or sets a value indicating whether the Prometheus metrics exporter is enabled.
    /// When <c>true</c>, exposes metrics in Prometheus format for scraping by Prometheus servers.
    /// </summary>
    /// <value><c>true</c> to enable Prometheus export; otherwise, <c>false</c>. Default is <c>true</c>.</value>
    public bool EnablePrometheusExporter { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether console exporters are enabled for development/debugging.
    /// When <c>true</c>, outputs metrics and traces to the console for local development.
    /// </summary>
    /// <value><c>true</c> to enable console export; otherwise, <c>false</c>. Default is <c>false</c>.</value>
    public bool EnableConsoleExporter { get; set; } = false;

    /// <summary>
    /// Gets or sets the OTLP (OpenTelemetry Protocol) collector endpoint URL.
    /// When specified, exports both metrics and traces to the OTLP collector.
    /// </summary>
    /// <value>The OTLP endpoint URL, or <c>null</c> to disable OTLP export.</value>
    /// <example>http://otel-collector:4317 or https://api.honeycomb.io</example>
    public string? OtlpEndpoint { get; set; }

    /// <summary>
    /// Gets or sets the protocol used for OTLP export (gRPC or HTTP/protobuf).
    /// This should match the protocol supported by the target OTLP collector.
    /// </summary>
    /// <value>The OTLP export protocol. Default is gRPC.</value>
    public OpenTelemetry.Exporter.OtlpExportProtocol OtlpProtocol { get; set; } = OpenTelemetry.Exporter.OtlpExportProtocol.Grpc;

    /// <summary>
    /// Gets or sets the Jaeger agent endpoint URL for distributed tracing.
    /// When specified, exports trace data to Jaeger for visualization and analysis.
    /// </summary>
    /// <value>The Jaeger endpoint URL, or <c>null</c> to disable Jaeger export.</value>
    /// <example>http://jaeger-agent:14268</example>
    public string? JaegerEndpoint { get; set; }

    #endregion
}