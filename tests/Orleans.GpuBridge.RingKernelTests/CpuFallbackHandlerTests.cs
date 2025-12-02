// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;
using Orleans.GpuBridge.Backends.DotCompute.RingKernels.CpuFallbackHandlers;
using Xunit;

namespace Orleans.GpuBridge.RingKernelTests;

/// <summary>
/// Tests for the CPU fallback handler system.
/// </summary>
public class CpuFallbackHandlerTests
{
    #region Registry Tests

    [Fact]
    public void Registry_RegisterHandler_ShouldSucceed()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();

        // Act
        var result = registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel",
            0,
            (request, state) => (new TestResponse { Result = request.Value * 2 }, state),
            "Test handler");

        // Assert
        Assert.True(result);
        Assert.Equal(1, registry.HandlerCount);
        Assert.True(registry.HasHandler("test_kernel", 0));
    }

    [Fact]
    public void Registry_RegisterDuplicateHandler_ShouldFail()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();
        registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel", 0, (req, state) => (default, state));

        // Act
        var result = registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel", 0, (req, state) => (default, state));

        // Assert
        Assert.False(result);
        Assert.Equal(1, registry.HandlerCount);
    }

    [Fact]
    public void Registry_ExecuteHandler_ShouldReturnCorrectResult()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();
        registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel",
            0,
            (request, state) => (new TestResponse { Result = request.Value + state.Counter },
                                 new TestState { Counter = state.Counter + 1 }));

        var request = new TestRequest { Value = 10 };
        var state = new TestState { Counter = 5 };

        // Act
        var result = registry.ExecuteHandler<TestRequest, TestResponse, TestState>(
            "test_kernel", 0, request, state);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(15, result.Value.Response.Result); // 10 + 5
        Assert.Equal(6, result.Value.NewState.Counter); // 5 + 1
    }

    [Fact]
    public void Registry_ExecuteHandler_UnregisteredKernel_ShouldReturnNull()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();

        // Act
        var result = registry.ExecuteHandler<TestRequest, TestResponse, TestState>(
            "unknown_kernel", 0, new TestRequest(), new TestState());

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void Registry_RemoveHandler_ShouldSucceed()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();
        registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel", 0, (req, state) => (default, state));

        // Act
        var result = registry.RemoveHandler("test_kernel", 0);

        // Assert
        Assert.True(result);
        Assert.Equal(0, registry.HandlerCount);
        Assert.False(registry.HasHandler("test_kernel", 0));
    }

    [Fact]
    public void Registry_Clear_ShouldRemoveAllHandlers()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();
        registry.RegisterHandler<TestRequest, TestResponse, TestState>("k1", 0, (r, s) => (default, s));
        registry.RegisterHandler<TestRequest, TestResponse, TestState>("k2", 0, (r, s) => (default, s));
        registry.RegisterHandler<TestRequest, TestResponse, TestState>("k3", 0, (r, s) => (default, s));

        // Act
        registry.Clear();

        // Assert
        Assert.Equal(0, registry.HandlerCount);
    }

    [Fact]
    public void Registry_RegisteredHandlers_ShouldReturnHandlerInfo()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();
        registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel", 0, (r, s) => (default, s), "Test handler description");

        // Act
        var handlers = registry.RegisteredHandlers;

        // Assert
        Assert.Single(handlers);
        var handler = handlers.First();
        Assert.Equal("test_kernel", handler.KernelId);
        Assert.Equal(0, handler.HandlerId);
        Assert.Equal("Test handler description", handler.Description);
    }

    [Fact]
    public void Registry_InvocationCount_ShouldTrackExecutions()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();
        registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel", 0, (r, s) => (default, s));

        // Act
        registry.ExecuteHandler<TestRequest, TestResponse, TestState>("test_kernel", 0, default, default);
        registry.ExecuteHandler<TestRequest, TestResponse, TestState>("test_kernel", 0, default, default);
        registry.ExecuteHandler<TestRequest, TestResponse, TestState>("test_kernel", 0, default, default);

        // Assert
        Assert.Equal(3, registry.GetInvocationCount("test_kernel", 0));
    }

    #endregion

    #region VectorAdd Handler Tests

    [Fact]
    public void VectorAddHandler_AddOperation_ShouldComputeCorrectly()
    {
        // Arrange
        var handler = new VectorAddCpuHandler();
        var request = new VectorAddProcessorRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = 0, // Add
            VectorLength = 4,
            A0 = 1.0f, A1 = 2.0f, A2 = 3.0f, A3 = 4.0f,
            B0 = 5.0f, B1 = 6.0f, B2 = 7.0f, B3 = 8.0f
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(6.0f, response.R0);
        Assert.Equal(8.0f, response.R1);
        Assert.Equal(10.0f, response.R2);
        Assert.Equal(12.0f, response.R3);
    }

    [Fact]
    public void VectorAddHandler_SubtractOperation_ShouldComputeCorrectly()
    {
        // Arrange
        var handler = new VectorAddCpuHandler();
        var request = new VectorAddProcessorRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = 1, // Subtract
            VectorLength = 4,
            A0 = 10.0f, A1 = 20.0f, A2 = 30.0f, A3 = 40.0f,
            B0 = 1.0f, B1 = 2.0f, B2 = 3.0f, B3 = 4.0f
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(9.0f, response.R0);
        Assert.Equal(18.0f, response.R1);
        Assert.Equal(27.0f, response.R2);
        Assert.Equal(36.0f, response.R3);
    }

    [Fact]
    public void VectorAddHandler_MultiplyOperation_ShouldComputeCorrectly()
    {
        // Arrange
        var handler = new VectorAddCpuHandler();
        var request = new VectorAddProcessorRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = 2, // Multiply
            VectorLength = 4,
            A0 = 2.0f, A1 = 3.0f, A2 = 4.0f, A3 = 5.0f,
            B0 = 3.0f, B1 = 4.0f, B2 = 5.0f, B3 = 6.0f
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(6.0f, response.R0);
        Assert.Equal(12.0f, response.R1);
        Assert.Equal(20.0f, response.R2);
        Assert.Equal(30.0f, response.R3);
    }

    [Fact]
    public void VectorAddHandler_DivideOperation_ShouldComputeCorrectly()
    {
        // Arrange
        var handler = new VectorAddCpuHandler();
        var request = new VectorAddProcessorRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = 3, // Divide
            VectorLength = 4,
            A0 = 10.0f, A1 = 20.0f, A2 = 30.0f, A3 = 40.0f,
            B0 = 2.0f, B1 = 4.0f, B2 = 5.0f, B3 = 8.0f
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(5.0f, response.R0);
        Assert.Equal(5.0f, response.R1);
        Assert.Equal(6.0f, response.R2);
        Assert.Equal(5.0f, response.R3);
    }

    [Fact]
    public void VectorAddHandler_DivideByZero_ShouldReturnZero()
    {
        // Arrange
        var handler = new VectorAddCpuHandler();
        var request = new VectorAddProcessorRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = 3, // Divide
            VectorLength = 4,
            A0 = 10.0f, A1 = 20.0f, A2 = 30.0f, A3 = 40.0f,
            B0 = 0.0f, B1 = 0.0f, B2 = 0.0f, B3 = 0.0f
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(0.0f, response.R0);
        Assert.Equal(0.0f, response.R1);
        Assert.Equal(0.0f, response.R2);
        Assert.Equal(0.0f, response.R3);
    }

    #endregion

    #region PatternMatch Handler Tests

    [Fact]
    public void PatternMatchHandler_MatchByProperty_ShouldFindMatches()
    {
        // Arrange
        var handler = new PatternMatchCpuHandler();
        var request = new PatternMatchRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = (int)PatternMatchOperation.MatchByProperty,
            VertexCount = 4,
            ComparisonOp = 0, // Equal
            PropertyValue = 5.0f,
            V0Property = 5.0f, // Match
            V1Property = 3.0f, // No match
            V2Property = 5.0f, // Match
            V3Property = 7.0f  // No match
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(2, response.MatchCount);
    }

    [Fact]
    public void PatternMatchHandler_MatchByDegree_ShouldFindMatches()
    {
        // Arrange
        var handler = new PatternMatchCpuHandler();
        var request = new PatternMatchRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = (int)PatternMatchOperation.MatchByDegree,
            VertexCount = 4,
            TargetDegree = 2,
            V0NeighborCount = 2, // Match
            V1NeighborCount = 3, // No match
            V2NeighborCount = 2, // Match
            V3NeighborCount = 1  // No match
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(2, response.MatchCount);
    }

    [Fact]
    public void PatternMatchHandler_MatchTriangle_ShouldDetectTriangle()
    {
        // Arrange - Create a triangle: 0-1, 1-2, 0-2
        var handler = new PatternMatchCpuHandler();
        var request = new PatternMatchRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = (int)PatternMatchOperation.MatchTriangle,
            VertexCount = 3,
            // V0 connected to V1, V2
            V0NeighborCount = 2,
            V0N0 = 1, V0N1 = 2,
            // V1 connected to V0, V2
            V1NeighborCount = 2,
            V1N0 = 0, V1N1 = 2,
            // V2 connected to V0, V1
            V2NeighborCount = 2,
            V2N0 = 0, V2N1 = 1
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.Equal(1, response.TriangleCount);
        Assert.Equal(3, response.MatchCount); // 3 vertices in triangle
    }

    [Fact]
    public void PatternMatchHandler_MatchStar_ShouldDetectStar()
    {
        // Arrange - Create a star with hub V0 connected to V1, V2, V3
        var handler = new PatternMatchCpuHandler();
        var request = new PatternMatchRingRequest
        {
            MessageId = Guid.NewGuid(),
            OperationType = (int)PatternMatchOperation.MatchStar,
            VertexCount = 4,
            TargetDegree = 3,
            // V0 is hub, connected to V1, V2, V3
            V0NeighborCount = 3,
            V0N0 = 1, V0N1 = 2, V0N2 = 3,
            // Spokes have degree 1
            V1NeighborCount = 1, V1N0 = 0,
            V2NeighborCount = 1, V2N0 = 0,
            V3NeighborCount = 1, V3N0 = 0
        };

        // Act
        var response = handler.Execute(request);

        // Assert
        Assert.True(response.Success);
        Assert.True(response.MatchCount >= 4); // Hub + spokes
    }

    #endregion

    #region Test Types

    private struct TestRequest
    {
        public int Value;
    }

    private struct TestResponse
    {
        public int Result;
    }

    private struct TestState
    {
        public int Counter;
    }

    #endregion
}
