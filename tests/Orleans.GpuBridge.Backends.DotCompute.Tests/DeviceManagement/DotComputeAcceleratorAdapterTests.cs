// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using DotCompute.Core.Compute;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Xunit;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.DeviceManagement;

/// <summary>
/// Integration tests for DotComputeAcceleratorAdapter with real DotCompute devices
/// </summary>
/// <remarks>
/// These tests use real DotCompute devices discovered on the system.
/// Tests will be skipped if no devices are available.
/// </remarks>
public class DotComputeAcceleratorAdapterTests
{
    private async Task<DotComputeAcceleratorAdapter?> GetFirstAvailableAdapter()
    {
        var manager = await DefaultAcceleratorManagerFactory.CreateAsync();
        var accelerators = await manager.GetAcceleratorsAsync();
        var firstAccelerator = accelerators.FirstOrDefault();

        if (firstAccelerator == null)
            return null;

        return new DotComputeAcceleratorAdapter(firstAccelerator, 0, NullLogger.Instance);
    }
    [Fact]
    public async Task Adapter_Should_MapDeviceType()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.Type.Should().BeOneOf(DeviceType.GPU, DeviceType.CPU);
    }

    [Fact]
    public async Task Adapter_Should_GenerateCorrectDeviceIdPattern()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.DeviceId.Should().StartWith("dotcompute-");
        adapter.DeviceId.Should().Contain("-0"); // Index 0
        adapter.Index.Should().Be(0);
    }

    [Fact]
    public async Task Adapter_Should_MapDeviceName()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.Name.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task Adapter_Should_MapArchitecture()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.Architecture.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task Adapter_Should_MapWarpSize()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.WarpSize.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task Adapter_Should_MapComputeUnits()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.ComputeUnits.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task Adapter_Should_MapTotalMemory()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.TotalMemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task Adapter_Should_MapComputeCapability()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.ComputeCapability.Should().NotBeNull();
        adapter.ComputeCapability.Major.Should().BeGreaterThanOrEqualTo(0);
        adapter.ComputeCapability.Minor.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task Adapter_Should_ExposeExtensionsViaProperties()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act
        var props = adapter.Properties;

        // Assert
        props.Should().ContainKey("extensions");
        props.Should().ContainKey("device_type");
        props.Should().ContainKey("compute_units");
    }

    [Fact]
    public async Task Adapter_Should_ReportHealthy_Initially()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        adapter.IsHealthy.Should().BeTrue();
        adapter.LastError.Should().BeNull();
    }

    [Fact]
    public async Task Adapter_Should_CalculateAvailableMemory()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act & Assert
        // Available memory should be less than or equal to total (DotCompute uses 80% heuristic)
        adapter.AvailableMemoryBytes.Should().BeGreaterThan(0);
        adapter.AvailableMemoryBytes.Should().BeLessThanOrEqualTo(adapter.TotalMemoryBytes);
    }

    [Fact]
    public async Task Adapter_Should_ReportDeviceStatus()
    {
        // Arrange
        var adapter = await GetFirstAvailableAdapter();
        if (adapter == null)
            return; // Skip if no devices

        // Act
        var status = adapter.GetStatus();

        // Assert
        status.Should().BeOneOf(DeviceStatus.Available, DeviceStatus.Busy, DeviceStatus.Error);
    }
}
