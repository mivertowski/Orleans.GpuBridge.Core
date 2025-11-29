// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.ResourceManagement;

/// <summary>
/// Resource request from a tenant
/// </summary>
public sealed class ResourceRequest
{
    /// <summary>
    /// Gets or sets the amount of GPU memory requested in bytes.
    /// </summary>
    public long RequestedMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the number of GPU kernels requested for execution.
    /// </summary>
    public int RequestedKernels { get; set; }

    /// <summary>
    /// Gets or sets the size of the batch to be processed by the kernels.
    /// </summary>
    public int BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the estimated duration for completing the resource request.
    /// </summary>
    public TimeSpan EstimatedDuration { get; set; }
}
