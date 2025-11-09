// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.Persistent;

/// <summary>
/// Options for persistent kernel execution
/// </summary>
public sealed class PersistentKernelOptions
{
    public int BatchSize { get; set; } = 100;
    public TimeSpan MaxBatchWaitTime { get; set; } = TimeSpan.FromMilliseconds(100);
    public bool RestartOnError { get; set; } = true;
    public int MaxRetries { get; set; } = 3;
    public TimeSpan RetryDelay { get; set; } = TimeSpan.FromSeconds(1);
}
