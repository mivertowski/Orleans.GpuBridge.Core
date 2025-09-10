using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Logging.Abstractions;
using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;

namespace Orleans.GpuBridge.Logging.Delegates;

/// <summary>
/// Logger delegate that writes to files with automatic rotation and retention.
/// </summary>
public sealed class FileLoggerDelegate : IBatchLoggerDelegate, IStructuredLoggerDelegate, IAsyncDisposable
{
    private readonly FileLoggerOptions _options;
    private readonly SemaphoreSlim _writeSemaphore = new(1, 1);
    private readonly Timer _rotationTimer;
    private string _currentLogFile;
    private FileStream? _currentStream;
    private StreamWriter? _currentWriter;
    private long _currentFileSize;
    private bool _disposed;

    public string Name => "File";
    public LogLevel MinimumLevel { get; }
    public int MaxBatchSize => _options.MaxBatchSize;

    public FileLoggerDelegate(FileLoggerOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        MinimumLevel = _options.MinimumLevel;
        
        // Ensure directory exists
        Directory.CreateDirectory(_options.LogDirectory);
        
        // Initialize current log file
        _currentLogFile = GetLogFileName();
        InitializeLogFile();
        
        // Setup rotation timer if needed
        _rotationTimer = _options.RotationInterval.HasValue 
            ? new Timer(RotateLogFile, null, _options.RotationInterval.Value, _options.RotationInterval.Value)
            : new Timer(_ => { }, null, Timeout.Infinite, Timeout.Infinite);
    }

    public bool IsEnabled(LogLevel logLevel) => logLevel >= MinimumLevel;

    public async Task WriteAsync(LogEntry entry, CancellationToken cancellationToken = default)
    {
        if (!IsEnabled(entry.Level) || _disposed)
            return;

        var enrichedEntry = EnrichEntry(entry);
        var formattedLine = FormatLogEntry(enrichedEntry);

        await _writeSemaphore.WaitAsync(cancellationToken);
        try
        {
            await WriteToCurrentFile(formattedLine, cancellationToken);
            await CheckRotationNeeded();
        }
        finally
        {
            _writeSemaphore.Release();
        }
    }

    public async Task WriteBatchAsync(IReadOnlyCollection<LogEntry> entries, CancellationToken cancellationToken = default)
    {
        if (entries.Count == 0 || _disposed)
            return;

        var filteredEntries = entries.Where(e => IsEnabled(e.Level)).ToList();
        if (filteredEntries.Count == 0)
            return;

        var lines = new List<string>(filteredEntries.Count);
        foreach (var entry in filteredEntries)
        {
            var enrichedEntry = EnrichEntry(entry);
            lines.Add(FormatLogEntry(enrichedEntry));
        }

        await _writeSemaphore.WaitAsync(cancellationToken);
        try
        {
            foreach (var line in lines)
            {
                await WriteToCurrentFile(line, cancellationToken);
            }
            await CheckRotationNeeded();
        }
        finally
        {
            _writeSemaphore.Release();
        }
    }

    public LogEntry EnrichEntry(LogEntry entry, LogContext? context = null)
    {
        var enrichedProperties = new Dictionary<string, object?>(entry.Properties);

        // Add file-specific metadata
        enrichedProperties["LogFile"] = Path.GetFileName(_currentLogFile);
        enrichedProperties["FileSize"] = _currentFileSize;

        // Add context properties if available
        context ??= LogContext.Current;
        if (context != null && _options.IncludeContext)
        {
            enrichedProperties["Machine"] = context.MachineName;
            enrichedProperties["Process"] = context.ProcessId;
            if (context.UserId != null) enrichedProperties["User"] = context.UserId;
            if (context.Component != null) enrichedProperties["Component"] = context.Component;
            if (context.Environment != null) enrichedProperties["Environment"] = context.Environment;
        }

        return entry.WithProperties(enrichedProperties);
    }

    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return;

        await _writeSemaphore.WaitAsync(cancellationToken);
        try
        {
            if (_currentWriter != null)
            {
                await _currentWriter.FlushAsync(cancellationToken);
                await _currentStream?.FlushAsync(cancellationToken)!;
            }
        }
        finally
        {
            _writeSemaphore.Release();
        }
    }

    ValueTask IAsyncDisposable.DisposeAsync() => DisposeAsync(CancellationToken.None);

    public async ValueTask DisposeAsync(CancellationToken cancellationToken = default)
    {
        if (_disposed)
            return;

        _disposed = true;

        await _rotationTimer.DisposeAsync();
        
        await _writeSemaphore.WaitAsync(cancellationToken);
        try
        {
            if (_currentWriter != null)
            {
                await _currentWriter.DisposeAsync();
                _currentWriter = null;
            }
            
            if (_currentStream != null)
            {
                await _currentStream.DisposeAsync();
                _currentStream = null;
            }
        }
        finally
        {
            _writeSemaphore.Release();
            _writeSemaphore.Dispose();
        }
    }

    private async Task WriteToCurrentFile(string line, CancellationToken cancellationToken)
    {
        if (_currentWriter == null)
            InitializeLogFile();

        await _currentWriter!.WriteLineAsync(line.AsMemory(), cancellationToken);
        _currentFileSize += Encoding.UTF8.GetByteCount(line) + Environment.NewLine.Length;

        if (_options.AutoFlush)
        {
            await _currentWriter.FlushAsync(cancellationToken);
        }
    }

    private async Task CheckRotationNeeded()
    {
        var needsRotation = false;

        // Check size-based rotation
        if (_options.MaxFileSizeBytes.HasValue && _currentFileSize >= _options.MaxFileSizeBytes.Value)
        {
            needsRotation = true;
        }

        // Check age-based rotation (handled by timer, but check here too)
        if (_options.RotationInterval.HasValue)
        {
            var fileInfo = new FileInfo(_currentLogFile);
            if (fileInfo.Exists && DateTime.UtcNow - fileInfo.CreationTimeUtc >= _options.RotationInterval.Value)
            {
                needsRotation = true;
            }
        }

        if (needsRotation)
        {
            await RotateLogFileAsync();
        }
    }

    private void InitializeLogFile()
    {
        _currentStream?.Dispose();
        _currentWriter?.Dispose();

        _currentStream = new FileStream(_currentLogFile, FileMode.Append, FileAccess.Write, FileShare.Read,
            bufferSize: _options.BufferSize, useAsync: true);
        _currentWriter = new StreamWriter(_currentStream, Encoding.UTF8);
        
        _currentFileSize = _currentStream.Length;
    }

    private string GetLogFileName()
    {
        var timestamp = DateTime.UtcNow.ToString(_options.FileNameTimestampFormat);
        var fileName = $"{_options.BaseFileName}_{timestamp}.log";
        return Path.Combine(_options.LogDirectory, fileName);
    }

    private void RotateLogFile(object? state)
    {
        Task.Run(async () =>
        {
            try
            {
                await RotateLogFileAsync();
            }
            catch (Exception ex)
            {
                // Log to console as fallback since file logging might be broken
                Console.WriteLine($"Failed to rotate log file: {ex.Message}");
            }
        });
    }

    private async Task RotateLogFileAsync()
    {
        await _writeSemaphore.WaitAsync();
        try
        {
            // Close current file
            if (_currentWriter != null)
            {
                await _currentWriter.DisposeAsync();
                _currentWriter = null;
            }
            if (_currentStream != null)
            {
                await _currentStream.DisposeAsync();
                _currentStream = null;
            }

            // Create new log file
            _currentLogFile = GetLogFileName();
            InitializeLogFile();

            // Clean up old files if retention is configured
            await CleanupOldFiles();
        }
        finally
        {
            _writeSemaphore.Release();
        }
    }

    private async Task CleanupOldFiles()
    {
        if (!_options.RetentionDays.HasValue && !_options.MaxRetainedFiles.HasValue)
            return;

        await Task.Run(() =>
        {
            try
            {
                var logFiles = Directory.GetFiles(_options.LogDirectory, $"{_options.BaseFileName}_*.log")
                    .Select(f => new FileInfo(f))
                    .Where(fi => fi.Exists)
                    .OrderByDescending(fi => fi.CreationTime)
                    .ToList();

                var filesToDelete = new List<FileInfo>();

                // Apply retention by age
                if (_options.RetentionDays.HasValue)
                {
                    var cutoffDate = DateTime.UtcNow.AddDays(-_options.RetentionDays.Value);
                    filesToDelete.AddRange(logFiles.Where(fi => fi.CreationTimeUtc < cutoffDate));
                }

                // Apply retention by count (keep the newest files)
                if (_options.MaxRetainedFiles.HasValue)
                {
                    var filesToKeep = logFiles.Take(_options.MaxRetainedFiles.Value).ToHashSet();
                    filesToDelete.AddRange(logFiles.Except(filesToKeep));
                }

                // Delete files
                foreach (var fileToDelete in filesToDelete.Distinct())
                {
                    try
                    {
                        fileToDelete.Delete();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to delete old log file {fileToDelete.FullName}: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to cleanup old log files: {ex.Message}");
            }
        });
    }

    private string FormatLogEntry(LogEntry entry)
    {
        var json = new
        {
            timestamp = entry.Timestamp.ToString("O"),
            level = entry.Level.ToString(),
            category = entry.Category,
            message = entry.Message,
            exception = entry.Exception?.ToString(),
            eventId = entry.EventId.Id != 0 ? entry.EventId.Id : (int?)null,
            eventName = !string.IsNullOrEmpty(entry.EventId.Name) ? entry.EventId.Name : null,
            correlationId = entry.CorrelationId,
            operationId = entry.OperationId,
            activityId = entry.ActivityId,
            threadId = entry.ThreadId,
            properties = entry.Properties.Count > 0 ? entry.Properties : null,
            scopes = entry.Scopes.Count > 0 ? entry.Scopes.Select(s => new { s.Name, s.Properties }).ToArray() : null,
            metrics = entry.Metrics != null ? new
            {
                duration = entry.Metrics.Duration?.TotalMilliseconds,
                memoryUsage = entry.Metrics.MemoryUsage,
                cpuUsage = entry.Metrics.CpuUsage,
                counters = entry.Metrics.Counters.Count > 0 ? entry.Metrics.Counters : null
            } : null
        };

        return JsonSerializer.Serialize(json, _options.JsonOptions);
    }
}

/// <summary>
/// Configuration options for file logging.
/// </summary>
public sealed class FileLoggerOptions
{
    /// <summary>
    /// Directory where log files will be stored.
    /// </summary>
    public string LogDirectory { get; set; } = "logs";

    /// <summary>
    /// Base name for log files (timestamp will be appended).
    /// </summary>
    public string BaseFileName { get; set; } = "app";

    /// <summary>
    /// Minimum log level to write to file.
    /// </summary>
    public LogLevel MinimumLevel { get; set; } = LogLevel.Debug;

    /// <summary>
    /// Maximum file size in bytes before rotation.
    /// </summary>
    public long? MaxFileSizeBytes { get; set; } = 100 * 1024 * 1024; // 100MB

    /// <summary>
    /// Time interval for file rotation.
    /// </summary>
    public TimeSpan? RotationInterval { get; set; } = TimeSpan.FromDays(1);

    /// <summary>
    /// Number of days to retain log files.
    /// </summary>
    public int? RetentionDays { get; set; } = 30;

    /// <summary>
    /// Maximum number of log files to retain.
    /// </summary>
    public int? MaxRetainedFiles { get; set; } = 100;

    /// <summary>
    /// Whether to flush after each write.
    /// </summary>
    public bool AutoFlush { get; set; } = false;

    /// <summary>
    /// Buffer size for file operations.
    /// </summary>
    public int BufferSize { get; set; } = 64 * 1024; // 64KB

    /// <summary>
    /// Maximum batch size for batch writes.
    /// </summary>
    public int MaxBatchSize { get; set; } = 1000;

    /// <summary>
    /// Timestamp format for file names.
    /// </summary>
    public string FileNameTimestampFormat { get; set; } = "yyyyMMdd_HHmmss";

    /// <summary>
    /// Whether to include context information.
    /// </summary>
    public bool IncludeContext { get; set; } = true;

    /// <summary>
    /// JSON serialization options.
    /// </summary>
    public JsonSerializerOptions JsonOptions { get; set; } = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };
}