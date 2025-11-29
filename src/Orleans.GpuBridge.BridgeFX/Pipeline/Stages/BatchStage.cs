using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.BridgeFX.Pipeline.Core;

namespace Orleans.GpuBridge.BridgeFX.Pipeline.Stages;

/// <summary>
/// Batch collection stage
/// </summary>
internal sealed class BatchStage<T> : IPipelineStage
    where T : notnull
{
    private readonly int _batchSize;
    private readonly TimeSpan _timeout;
    private readonly List<T> _buffer = new();
    private DateTime _lastFlush = DateTime.UtcNow;

    public Type InputType => typeof(T);
    public Type OutputType => typeof(IReadOnlyList<T>);

    public BatchStage(int batchSize, TimeSpan timeout)
    {
        _batchSize = batchSize;
        _timeout = timeout;
    }

    public Task<object?> ProcessAsync(object input, CancellationToken ct)
    {
        if (input is not T typedInput)
        {
            throw new ArgumentException($"Expected {typeof(T)}, got {input.GetType()}");
        }

        lock (_buffer)
        {
            _buffer.Add(typedInput);

            var shouldFlush = _buffer.Count >= _batchSize ||
                             DateTime.UtcNow - _lastFlush >= _timeout;

            if (shouldFlush)
            {
                var batch = _buffer.ToList();
                _buffer.Clear();
                _lastFlush = DateTime.UtcNow;
                return Task.FromResult<object?>(batch);
            }
        }

        return Task.FromResult<object?>(null);
    }
}