// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;

namespace Orleans.GpuBridge.Runtime.ResourceManagement;

/// <summary>
/// Resource request from a tenant
/// </summary>
public sealed class ResourceRequest
{
    public long RequestedMemoryBytes { get; set; }
    public int RequestedKernels { get; set; }
    public int BatchSize { get; set; }
    public TimeSpan EstimatedDuration { get; set; }
}
