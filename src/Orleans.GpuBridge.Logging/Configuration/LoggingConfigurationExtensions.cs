using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Logging.Configuration;

/// <summary>
/// Configuration extensions for Orleans GPU Bridge logging.
/// </summary>
public static class LoggingConfigurationExtensions
{
    /// <summary>
    /// Binds configuration to GpuBridgeLoggingOptions.
    /// </summary>
    public static GpuBridgeLoggingOptions BindGpuBridgeLogging(this IConfiguration configuration)
    {
        var options = new GpuBridgeLoggingOptions();
        configuration.GetSection(GpuBridgeLoggingOptions.SectionName).Bind(options);
        return options;
    }

    /// <summary>
    /// Gets a specific category log level from configuration.
    /// </summary>
    public static LogLevel GetCategoryLevel(this GpuBridgeLoggingOptions options, string categoryName)
    {
        // Check for exact match first
        if (options.CategoryLevels.TryGetValue(categoryName, out var level))
        {
            return level;
        }

        // Check for parent namespace matches
        var parts = categoryName.Split('.');
        for (int i = parts.Length - 1; i > 0; i--)
        {
            var parentCategory = string.Join('.', parts.Take(i));
            if (options.CategoryLevels.TryGetValue(parentCategory, out level))
            {
                return level;
            }
        }

        return options.MinimumLevel;
    }

    /// <summary>
    /// Validates configuration options.
    /// </summary>
    public static ValidationResult ValidateConfiguration(this GpuBridgeLoggingOptions options)
    {
        var errors = new List<string>();

        if (options.Buffer.Capacity <= 0)
            errors.Add("Buffer capacity must be greater than 0");

        if (options.Buffer.MaxBatchSize <= 0)
            errors.Add("Buffer max batch size must be greater than 0");

        if (options.DelegateManager.MaxQueueSize <= 0)
            errors.Add("Delegate manager max queue size must be greater than 0");

        if (options.File.Enabled)
        {
            if (string.IsNullOrWhiteSpace(options.File.LogDirectory))
                errors.Add("File log directory must be specified when file logging is enabled");

            if (string.IsNullOrWhiteSpace(options.File.BaseFileName))
                errors.Add("File base name must be specified when file logging is enabled");
        }

        if (options.Telemetry.Enabled)
        {
            if (string.IsNullOrWhiteSpace(options.Telemetry.ServiceName))
                errors.Add("Service name must be specified when telemetry is enabled");

            if (options.Telemetry.SamplingRate is < 0 or > 1)
                errors.Add("Telemetry sampling rate must be between 0 and 1");
        }

        return new ValidationResult(errors.Count == 0, errors);
    }
}
