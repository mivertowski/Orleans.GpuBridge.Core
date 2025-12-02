// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;

namespace Orleans.GpuBridge.Abstractions.Exceptions;

/// <summary>
/// Base exception class for all GPU Bridge exceptions.
/// </summary>
[Serializable]
public class GpuBridgeException : Exception
{
    /// <summary>
    /// Error code for categorizing the exception.
    /// </summary>
    public string ErrorCode { get; }

    /// <summary>
    /// Name of the operation that failed.
    /// </summary>
    public string? OperationName { get; }

    /// <summary>
    /// Additional context data about the exception.
    /// </summary>
    public object? Context { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuBridgeException"/> class.
    /// </summary>
    public GpuBridgeException()
        : this("GPU_BRIDGE_ERROR", "An error occurred in GPU Bridge.")
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuBridgeException"/> class with a message.
    /// </summary>
    /// <param name="message">The exception message.</param>
    public GpuBridgeException(string message)
        : this("GPU_BRIDGE_ERROR", message)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuBridgeException"/> class with a message and inner exception.
    /// </summary>
    /// <param name="message">The exception message.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuBridgeException(string message, Exception innerException)
        : this("GPU_BRIDGE_ERROR", message, null, null, innerException)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuBridgeException"/> class with an error code and message.
    /// </summary>
    /// <param name="errorCode">The error code.</param>
    /// <param name="message">The exception message.</param>
    public GpuBridgeException(string errorCode, string message)
        : this(errorCode, message, null, null, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuBridgeException"/> class with full details.
    /// </summary>
    /// <param name="errorCode">The error code.</param>
    /// <param name="message">The exception message.</param>
    /// <param name="operationName">The name of the operation that failed.</param>
    /// <param name="context">Additional context data.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuBridgeException(
        string errorCode,
        string message,
        string? operationName,
        object? context = null,
        Exception? innerException = null)
        : base(message, innerException)
    {
        ErrorCode = errorCode ?? "GPU_BRIDGE_ERROR";
        OperationName = operationName;
        Context = context;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var result = $"[{ErrorCode}] {Message}";
        if (!string.IsNullOrEmpty(OperationName))
        {
            result = $"{result} (Operation: {OperationName})";
        }
        if (InnerException != null)
        {
            result = $"{result}\n---> {InnerException}";
        }
        return result;
    }
}

/// <summary>
/// Exception thrown when a GPU operation fails.
/// </summary>
[Serializable]
public sealed class GpuOperationException : GpuBridgeException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GpuOperationException"/> class.
    /// </summary>
    /// <param name="operationName">The name of the operation that failed.</param>
    /// <param name="message">The exception message.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuOperationException(string operationName, string message, Exception? innerException = null)
        : base("GPU_OPERATION_FAILED", message, operationName, null, innerException)
    {
    }
}

/// <summary>
/// Exception thrown when GPU memory allocation fails.
/// </summary>
[Serializable]
public sealed class GpuMemoryException : GpuBridgeException
{
    /// <summary>
    /// The requested memory size in bytes.
    /// </summary>
    public long RequestedBytes { get; }

    /// <summary>
    /// The available memory in bytes.
    /// </summary>
    public long? AvailableBytes { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuMemoryException"/> class.
    /// </summary>
    /// <param name="message">The exception message.</param>
    /// <param name="requestedBytes">The requested memory size.</param>
    /// <param name="availableBytes">The available memory size.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuMemoryException(
        string message,
        long requestedBytes,
        long? availableBytes = null,
        Exception? innerException = null)
        : base("GPU_MEMORY_ALLOCATION_FAILED", message, "MemoryAllocation",
            new { RequestedBytes = requestedBytes, AvailableBytes = availableBytes }, innerException)
    {
        RequestedBytes = requestedBytes;
        AvailableBytes = availableBytes;
    }
}

/// <summary>
/// Exception thrown when a kernel execution fails.
/// </summary>
[Serializable]
public sealed class KernelExecutionException : GpuBridgeException
{
    /// <summary>
    /// The kernel ID that failed.
    /// </summary>
    public string? KernelId { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelExecutionException"/> class.
    /// </summary>
    /// <param name="kernelId">The kernel ID.</param>
    /// <param name="message">The exception message.</param>
    /// <param name="innerException">The inner exception.</param>
    public KernelExecutionException(string? kernelId, string message, Exception? innerException = null)
        : base("KERNEL_EXECUTION_FAILED", message, "KernelExecution", new { KernelId = kernelId }, innerException)
    {
        KernelId = kernelId;
    }
}

/// <summary>
/// Exception thrown when a GPU device is not available.
/// </summary>
[Serializable]
public sealed class GpuDeviceUnavailableException : GpuBridgeException
{
    /// <summary>
    /// The device index that was requested.
    /// </summary>
    public int? DeviceIndex { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuDeviceUnavailableException"/> class.
    /// </summary>
    /// <param name="message">The exception message.</param>
    /// <param name="deviceIndex">The device index.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuDeviceUnavailableException(string message, int? deviceIndex = null, Exception? innerException = null)
        : base("GPU_DEVICE_UNAVAILABLE", message, "DeviceAccess", new { DeviceIndex = deviceIndex }, innerException)
    {
        DeviceIndex = deviceIndex;
    }
}

/// <summary>
/// Exception thrown when a GPU kernel fails.
/// </summary>
[Serializable]
public sealed class GpuKernelException : GpuBridgeException
{
    /// <summary>
    /// The kernel ID that failed.
    /// </summary>
    public string? KernelId { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuKernelException"/> class.
    /// </summary>
    /// <param name="message">The exception message.</param>
    /// <param name="kernelId">The kernel ID.</param>
    /// <param name="operationName">The operation name.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuKernelException(string message, string? kernelId = null, string? operationName = null, Exception? innerException = null)
        : base("GPU_KERNEL_FAILED", message, operationName, new { KernelId = kernelId }, innerException)
    {
        KernelId = kernelId;
    }
}

/// <summary>
/// Exception thrown when a GPU device operation fails.
/// </summary>
[Serializable]
public sealed class GpuDeviceException : GpuBridgeException
{
    /// <summary>
    /// The device index.
    /// </summary>
    public int DeviceIndex { get; }

    /// <summary>
    /// The device name.
    /// </summary>
    public string? DeviceName { get; }

    /// <summary>
    /// The device state when the error occurred.
    /// </summary>
    public string? DeviceState { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuDeviceException"/> class.
    /// </summary>
    /// <param name="message">The exception message.</param>
    /// <param name="deviceIndex">The device index.</param>
    /// <param name="deviceName">The device name.</param>
    /// <param name="deviceState">The device state.</param>
    /// <param name="operationName">The operation name.</param>
    /// <param name="innerException">The inner exception.</param>
    public GpuDeviceException(
        string message,
        int deviceIndex,
        string? deviceName = null,
        string? deviceState = null,
        string? operationName = null,
        Exception? innerException = null)
        : base("GPU_DEVICE_ERROR", message, operationName,
            new { DeviceIndex = deviceIndex, DeviceName = deviceName, DeviceState = deviceState }, innerException)
    {
        DeviceIndex = deviceIndex;
        DeviceName = deviceName;
        DeviceState = deviceState;
    }
}

/// <summary>
/// Exception thrown when a rate limit is exceeded.
/// </summary>
[Serializable]
public sealed class RateLimitExceededException : GpuBridgeException
{
    /// <summary>
    /// The number of requests made.
    /// </summary>
    public int RequestCount { get; }

    /// <summary>
    /// The maximum allowed requests.
    /// </summary>
    public int MaxRequests { get; }

    /// <summary>
    /// The time until the limit resets.
    /// </summary>
    public TimeSpan RetryAfter { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="RateLimitExceededException"/> class.
    /// </summary>
    /// <param name="requestCount">The request count.</param>
    /// <param name="maxRequests">The maximum requests.</param>
    /// <param name="retryAfter">The retry after duration.</param>
    public RateLimitExceededException(int requestCount, int maxRequests, TimeSpan retryAfter)
        : base("RATE_LIMIT_EXCEEDED",
            $"Rate limit exceeded: {requestCount} requests (max: {maxRequests}). Retry after {retryAfter.TotalSeconds:F1} seconds.",
            "RateLimiting",
            new { RequestCount = requestCount, MaxRequests = maxRequests, RetryAfter = retryAfter })
    {
        RequestCount = requestCount;
        MaxRequests = maxRequests;
        RetryAfter = retryAfter;
    }
}
