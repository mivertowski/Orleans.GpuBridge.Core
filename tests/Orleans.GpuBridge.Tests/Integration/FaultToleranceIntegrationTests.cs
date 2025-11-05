using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains;
using Orleans.GpuBridge.Grains.Batch;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.Integration;

/// <summary>
/// Fault tolerance and recovery integration tests for Orleans GPU Bridge
/// </summary>
public class FaultToleranceIntegrationTests : IClassFixture<GpuClusterFixture>
{
    private readonly GpuClusterFixture _fixture;
    private readonly ITestOutputHelper _output;

    public FaultToleranceIntegrationTests(GpuClusterFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Fact]
    public async Task GrainFailure_ShouldRecoverGracefully()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(),"failure-recovery-test");
        
        var validInput = new[] { new float[] { 1.0f, 2.0f, 3.0f } };
        var invalidInput = new[] { new float[] { float.NaN, float.PositiveInfinity } };

        // Act & Assert - First execution should succeed
        var result1 = await grain.ExecuteAsync(validInput);
        result1.Success.Should().BeTrue();

        // Second execution with invalid data should be handled gracefully
        var result2 = await grain.ExecuteAsync(invalidInput);
        // Should not throw exception, either succeeds with fallback or returns error result
        result2.Should().NotBeNull();

        // Third execution should work normally (grain should recover)
        var result3 = await grain.ExecuteAsync(validInput);
        result3.Success.Should().BeTrue();
        
        _output.WriteLine($"Recovery test: {result1.Success}, {result2.Success}, {result3.Success}");
    }

    [Fact]
    public async Task HighErrorRate_ShouldMaintainSystemStability()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        const int totalOperations = 100;
        const int errorOperations = 30; // 30% error rate
        
        var tasks = new List<Task<GpuBatchResult<float[]>>>();
        var random = new Random();

        // Act - Generate mix of valid and invalid operations
        for (int i = 0; i < totalOperations; i++)
        {
            var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), Guid.NewGuid().ToString(), $"error-test-{i}");
            
            float[] input;
            if (i < errorOperations && random.NextDouble() < 0.5) // Randomly distribute errors
            {
                // Create problematic input
                input = new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity };
            }
            else
            {
                // Create valid input
                input = Enumerable.Range(0, 1000).Select(x => (float)x).ToArray();
            }
            
            tasks.Add(grain.ExecuteAsync(new[] { input }));
        }

        var results = await Task.WhenAll(tasks);

        // Assert - System should remain stable despite errors
        results.Should().HaveCount(totalOperations);
        results.Should().OnlyContain(r => r != null); // No null results
        
        var successCount = results.Count(r => r.Success);
        var errorCount = results.Count(r => !r.Success);
        
        _output.WriteLine($"Successful operations: {successCount}/{totalOperations}");
        _output.WriteLine($"Failed operations: {errorCount}/{totalOperations}");
        _output.WriteLine($"Success rate: {(double)successCount / totalOperations * 100:F1}%");

        // At least 50% should succeed (valid operations + some error recovery)
        successCount.Should().BeGreaterThan(totalOperations / 2);
    }

    [Fact]
    public async Task ConcurrentFailures_ShouldNotCascade()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        const int concurrentGrains = 20;
        
        var tasks = new List<Task<(string grainId, GpuBatchResult<float[]> result1, GpuBatchResult<float[]> result2)>>();

        // Act - Each grain performs one normal and one potentially problematic operation
        for (int i = 0; i < concurrentGrains; i++)
        {
            var grainId = $"cascade-test-{i}";
            tasks.Add(Task.Run(async () =>
            {
                var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), Guid.NewGuid().ToString(), Guid.NewGuid().ToString(), grainId);
                
                // Normal operation
                var normalInput = new[] { new float[] { i, i + 1, i + 2 } };
                var result1 = await grain.ExecuteAsync(normalInput);
                
                // Potentially problematic operation
                var problematicInput = new[] { new float[] { float.MaxValue, float.MinValue } };
                var result2 = await grain.ExecuteAsync(problematicInput);
                
                return (grainId, result1, result2);
            }));
        }

        var results = await Task.WhenAll(tasks);

        // Assert - Failures in some grains should not affect others
        results.Should().HaveCount(concurrentGrains);
        
        var normalOperationSuccesses = results.Count(r => r.result1.Success);
        var problematicOperationHandled = results.Count(r => r.result2 != null); // Should not throw
        
        _output.WriteLine($"Normal operations succeeded: {normalOperationSuccesses}/{concurrentGrains}");
        _output.WriteLine($"Problematic operations handled: {problematicOperationHandled}/{concurrentGrains}");

        // Most normal operations should succeed
        normalOperationSuccesses.Should().BeGreaterThan((int)(concurrentGrains * 0.8));
        // All problematic operations should be handled (not crash)
        problematicOperationHandled.Should().Be(concurrentGrains);
    }

    [Fact]
    public async Task MemoryPressure_ShouldHandleGracefully()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(),"memory-pressure-test");
        
        const int iterations = 20;
        const int largeVectorSize = 100000; // 100k elements per vector
        
        var successCount = 0;
        var errors = new List<string>();

        // Act - Apply memory pressure
        for (int i = 0; i < iterations; i++)
        {
            try
            {
                // Create large data sets
                var largeInput = Enumerable.Range(0, 10)
                    .Select(_ => Enumerable.Range(0, largeVectorSize)
                        .Select(x => (float)x)
                        .ToArray())
                    .ToArray();

                var result = await grain.ExecuteAsync(largeInput);
                
                if (result.Success)
                {
                    successCount++;
                }
                else if (result.Error != null)
                {
                    errors.Add(result.Error);
                }

                // Occasionally force garbage collection to test memory management
                if (i % 5 == 0)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
                
                _output.WriteLine($"Iteration {i + 1}/{iterations}: {(result.Success ? "Success" : "Failed")}");
            }
            catch (Exception ex)
            {
                errors.Add(ex.Message);
                _output.WriteLine($"Iteration {i + 1}/{iterations}: Exception - {ex.Message}");
            }
        }

        // Assert
        _output.WriteLine($"Successful iterations: {successCount}/{iterations}");
        _output.WriteLine($"Errors encountered: {errors.Count}");
        
        // Should handle memory pressure gracefully - at least 70% success rate
        successCount.Should().BeGreaterThan((int)(iterations * 0.7));
        
        // If there are errors, they should be handled gracefully (not crash the system)
        errors.ForEach(error => _output.WriteLine($"Error: {error}"));
    }

    [Fact]
    public async Task TimeoutHandling_ShouldNotBlockSystem()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        
        // Use a grain that might take longer to process
        var slowGrain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), "timeout-test-slow");
        var fastGrain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), "timeout-test-fast");

        var largeInput = Enumerable.Range(0, 10000)
            .Select(x => (float)x)
            .ToArray();
        var smallInput = new[] { 1.0f, 2.0f, 3.0f };

        // Act - Start slow operation and fast operation concurrently
        var slowTask = slowGrain.ExecuteAsync(new[] { largeInput });
        var fastTask = fastGrain.ExecuteAsync(new[] { smallInput });

        // Add timeout to slow task
        var timeoutTask = Task.Delay(TimeSpan.FromSeconds(30));
        var completedTask = await Task.WhenAny(slowTask, fastTask, timeoutTask);

        // Assert
        if (completedTask == fastTask)
        {
            var fastResult = await fastTask;
            fastResult.Success.Should().BeTrue();
            _output.WriteLine("Fast operation completed successfully");
        }
        else if (completedTask == slowTask)
        {
            var slowResult = await slowTask;
            _output.WriteLine($"Slow operation completed: {slowResult.Success}");
        }
        else
        {
            _output.WriteLine("Test timed out - this is acceptable for timeout testing");
        }

        // The key is that fast operations should not be blocked by slow ones
        var fastCompleted = fastTask.IsCompleted && !fastTask.IsFaulted;
        fastCompleted.Should().BeTrue("Fast operations should not be blocked by slow operations");
    }

    [Fact]
    public async Task ResourceExhaustion_ShouldFallbackGracefully()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        const int excessiveGrainCount = 100; // More grains than typical system can handle efficiently
        
        var tasks = new List<Task<GpuBatchResult<float[]>>>();
        var grains = new List<IGpuBatchGrain<float[], float[]>>();

        // Act - Create excessive number of concurrent operations
        for (int i = 0; i < excessiveGrainCount; i++)
        {
            var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), Guid.NewGuid().ToString(), $"exhaustion-test-{i}");
            grains.Add(grain);
            
            var input = new[] { Enumerable.Range(0, 1000).Select(x => (float)x).ToArray() };
            tasks.Add(grain.ExecuteAsync(input));
        }

        var results = await Task.WhenAll(tasks);

        // Assert
        results.Should().HaveCount(excessiveGrainCount);
        results.Should().OnlyContain(r => r != null);
        
        var successCount = results.Count(r => r.Success);
        var failureCount = results.Count(r => !r.Success);
        
        _output.WriteLine($"Resource exhaustion test: {successCount} successes, {failureCount} failures");
        
        // System should handle resource exhaustion gracefully
        // Either operations succeed (with potential CPU fallback) or fail gracefully
        (successCount + failureCount).Should().Be(excessiveGrainCount);
        
        // At least some operations should succeed
        successCount.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task PartialFailure_BatchProcessing_ShouldContinueProcessing()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(),"partial-failure-test");
        
        // Create batch with mix of valid and invalid data
        var batch = new List<float[]>
        {
            new float[] { 1.0f, 2.0f, 3.0f }, // Valid
            new float[] { float.NaN, float.PositiveInfinity }, // Invalid
            new float[] { 4.0f, 5.0f, 6.0f }, // Valid
            new float[] { float.NegativeInfinity }, // Invalid
            new float[] { 7.0f, 8.0f, 9.0f }  // Valid
        };

        // Act
        var result = await grain.ExecuteAsync(batch);

        // Assert
        result.Should().NotBeNull();
        _output.WriteLine($"Partial failure test: Success={result.Success}, Results count={result.Results.Count}");
        
        // The system should either:
        // 1. Process all items successfully (with data sanitization/fallback)
        // 2. Process valid items and handle invalid ones gracefully
        // 3. Return an error result but not crash
        
        if (result.Success)
        {
            result.Results.Should().HaveCount(batch.Count);
        }
        // If not successful, should still provide meaningful error information
        else
        {
            result.Error.Should().NotBeNullOrEmpty();
        }
    }

    [Fact]
    public async Task GrainReactivation_AfterFailure_ShouldRestoreState()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        var grainId = "reactivation-test";
        
        // Act - First operation
        var grain1 = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(grainId);
        var input = new[] { new float[] { 1.0f, 2.0f, 3.0f } };
        var result1 = await grain1.ExecuteAsync(input);
        
        // Simulate grain deactivation by waiting
        await Task.Delay(1000);
        
        // Force potential issue that might cause internal state problems
        var problematicInput = new[] { new float[] { float.MaxValue } };
        var result2 = await grain1.ExecuteAsync(problematicInput);
        
        // Wait again to allow for potential deactivation
        await Task.Delay(1000);
        
        // Third operation should work (grain should be in clean state)
        var grain2 = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(grainId);
        var result3 = await grain2.ExecuteAsync(input);

        // Assert
        result1.Success.Should().BeTrue("First operation should succeed");
        // result2 may succeed or fail, both are acceptable
        result3.Success.Should().BeTrue("Third operation should succeed after reactivation");
        
        _output.WriteLine($"Reactivation test: {result1.Success} -> {result2.Success} -> {result3.Success}");
    }

    [Fact]
    public async Task SystemRecovery_AfterMultipleFailures_ShouldStabilize()
    {
        // Arrange
        var grainFactory = _fixture.Cluster.GrainFactory;
        const int recoveryPhases = 3;
        const int operationsPerPhase = 20;
        
        var allResults = new List<(int phase, bool success)>();

        // Act - Multiple phases of operations with recovery periods
        for (int phase = 0; phase < recoveryPhases; phase++)
        {
            _output.WriteLine($"Starting recovery phase {phase + 1}");
            
            var phaseTasks = new List<Task<GpuBatchResult<float[]>>>();
            
            for (int op = 0; op < operationsPerPhase; op++)
            {
                var grain = grainFactory.GetGrain<IGpuBatchGrain<float[], float[]>>(Guid.NewGuid(), Guid.NewGuid().ToString(), Guid.NewGuid().ToString(), $"recovery-{phase}-{op}");
                
                // Mix of normal and challenging operations
                float[] input = op % 3 == 0 
                    ? new[] { float.MaxValue, float.MinValue } // Challenging
                    : Enumerable.Range(op, 100).Select(x => (float)x).ToArray(); // Normal
                
                phaseTasks.Add(grain.ExecuteAsync(new[] { input }));
            }
            
            var phaseResults = await Task.WhenAll(phaseTasks);
            
            foreach (var result in phaseResults)
            {
                allResults.Add((phase, result.Success));
            }
            
            // Recovery period between phases
            await Task.Delay(2000);
            GC.Collect(); // Help with cleanup
        }

        // Assert - System should show recovery over time
        var phaseSuccessRates = new double[recoveryPhases];
        
        for (int phase = 0; phase < recoveryPhases; phase++)
        {
            var phaseResults = allResults.Where(r => r.phase == phase).ToList();
            var successCount = phaseResults.Count(r => r.success);
            phaseSuccessRates[phase] = (double)successCount / phaseResults.Count;
            
            _output.WriteLine($"Phase {phase + 1} success rate: {phaseSuccessRates[phase] * 100:F1}%");
        }

        // Later phases should generally have equal or better success rates (system stabilization)
        phaseSuccessRates[recoveryPhases - 1].Should().BeGreaterThan(0.5, 
            "Final phase should have reasonable success rate indicating system recovery");
    }
}