// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.Persistent;

/// <summary>
/// Options for persistent kernel execution.
/// </summary>
public sealed class PersistentKernelOptions
{
    /// <summary>
    /// Gets or sets the number of items to process in each batch. Default is 100.
    /// </summary>
    public int BatchSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum time to wait before flushing a partial batch. Default is 100ms.
    /// </summary>
    public TimeSpan MaxBatchWaitTime { get; set; } = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Gets or sets whether to automatically restart the kernel on error. Default is true.
    /// </summary>
    public bool RestartOnError { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of restart retries. Default is 3.
    /// </summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>
    /// Gets or sets the delay between restart attempts. Default is 1 second.
    /// </summary>
    public TimeSpan RetryDelay { get; set; } = TimeSpan.FromSeconds(1);
}
