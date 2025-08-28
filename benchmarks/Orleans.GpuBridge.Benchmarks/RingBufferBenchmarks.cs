// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime.Persistent;

namespace Orleans.GpuBridge.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class RingBufferBenchmarks
{
    private RingBuffer _ringBuffer = null!;
    private byte[] _smallData = null!;
    private byte[] _mediumData = null!;
    private byte[] _largeData = null!;
    
    [Params(1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024)]
    public int BufferSize { get; set; }
    
    [GlobalSetup]
    public void Setup()
    {
        _ringBuffer = new RingBuffer("test-kernel", BufferSize, NullLogger.Instance);
        
        _smallData = new byte[1024]; // 1KB
        _mediumData = new byte[64 * 1024]; // 64KB
        _largeData = new byte[1024 * 1024]; // 1MB
        
        Random.Shared.NextBytes(_smallData);
        Random.Shared.NextBytes(_mediumData);
        Random.Shared.NextBytes(_largeData);
    }
    
    [GlobalCleanup]
    public void Cleanup()
    {
        _ringBuffer?.Dispose();
    }
    
    [Benchmark]
    public bool WriteSmallData()
    {
        return _ringBuffer.TryWrite(_smallData);
    }
    
    [Benchmark]
    public bool WriteMediumData()
    {
        return _ringBuffer.TryWrite(_mediumData);
    }
    
    [Benchmark]
    public bool WriteLargeData()
    {
        return _ringBuffer.TryWrite(_largeData);
    }
    
    [Benchmark]
    public Memory<byte>? WriteAndRead()
    {
        _ringBuffer.TryWrite(_smallData);
        return _ringBuffer.TryRead();
    }
    
    [Benchmark]
    public void BurstWrite()
    {
        for (int i = 0; i < 100; i++)
        {
            if (!_ringBuffer.TryWrite(_smallData))
                break;
        }
        
        // Drain buffer
        while (_ringBuffer.TryRead() != null)
        {
            // Just consume
        }
    }
    
    [Benchmark]
    public async Task ConcurrentWriteRead()
    {
        var writeTask = Task.Run(() =>
        {
            for (int i = 0; i < 1000; i++)
            {
                _ringBuffer.TryWrite(_smallData);
            }
        });
        
        var readTask = Task.Run(() =>
        {
            int readCount = 0;
            while (readCount < 1000)
            {
                if (_ringBuffer.TryRead() != null)
                    readCount++;
            }
        });
        
        await Task.WhenAll(writeTask, readTask);
    }
    
    [Benchmark]
    public RingBufferStats GetStatistics()
    {
        // Write some data first
        for (int i = 0; i < 10; i++)
        {
            _ringBuffer.TryWrite(_smallData);
        }
        
        return _ringBuffer.GetStats();
    }
}