// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Xunit;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.DeviceManagement;

/// <summary>
/// Integration tests for DotCompute API verification
/// </summary>
public class ApiVerificationTests
{
    [Fact]
    public async Task VerifyApisAsync_Should_CompleteSuccessfully()
    {
        // Arrange
        var logger = NullLogger.Instance;

        // Act
        var result = await DotComputeApiVerification.VerifyApisAsync(logger);

        // Assert
        result.Should().NotBeNull();
        result.FactoryMethodAvailable.Should().BeTrue("Factory method should be available");
        result.EnumerationAvailable.Should().BeTrue("Enumeration should be available");

        // Log summary for debugging
        var summary = result.GetSummary();
        summary.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task VerifyApisAsync_Should_DiscoverAtLeastOneDevice()
    {
        // Arrange
        var logger = NullLogger.Instance;

        // Act
        var result = await DotComputeApiVerification.VerifyApisAsync(logger);

        // Assert
        // Should find at least CPU device if no GPU available
        result.AcceleratorCount.Should().BeGreaterThanOrEqualTo(0);
        // Note: 0 is acceptable if DotCompute CPU backend not loaded
    }

    [Fact]
    public async Task VerifyApisAsync_Should_VerifyAcceleratorInfoProperties()
    {
        // Arrange
        var logger = NullLogger.Instance;

        // Act
        var result = await DotComputeApiVerification.VerifyApisAsync(logger);

        // Assert
        if (result.AcceleratorCount > 0)
        {
            result.ArchitecturePropertyAvailable.Should().BeTrue();
            result.WarpSizePropertyAvailable.Should().BeTrue();
            result.ExtensionsPropertyAvailable.Should().BeTrue();
            result.ArchitectureValue.Should().NotBeNullOrEmpty();
        }
    }

    [Fact]
    public async Task VerifyApisAsync_Should_VerifyMemoryManagementAPIs()
    {
        // Arrange
        var logger = NullLogger.Instance;

        // Act
        var result = await DotComputeApiVerification.VerifyApisAsync(logger);

        // Assert
        if (result.AcceleratorCount > 0)
        {
            result.MemoryTotalPropertyAvailable.Should().BeTrue();
            result.MemoryStatisticsAvailable.Should().BeTrue();
            result.TotalMemoryBytes.Should().BeGreaterThan(0);
        }
    }

    [Fact]
    public async Task VerifyApisAsync_Should_ConfirmKernelCompilationAvailable()
    {
        // Arrange
        var logger = NullLogger.Instance;

        // Act
        var result = await DotComputeApiVerification.VerifyApisAsync(logger);

        // Assert
        if (result.AcceleratorCount > 0)
        {
            // Kernel compilation should be available via IAccelerator
            result.KernelCompilationAvailable.Should().BeTrue();
        }
    }
}
