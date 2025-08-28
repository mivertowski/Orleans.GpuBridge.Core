// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Runtime;

namespace Orleans.GpuBridge.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[ThreadingDiagnoser]
public class MemoryPoolBenchmarks
{
    private AdvancedMemoryPool<float> _floatPool = null!;
    private AdvancedMemoryPool<byte> _bytePool = null!;
    private MemoryPoolManager _poolManager = null!;
    
    [Params(1024, 4096, 16384, 65536)]
    public int BufferSize { get; set; }
    
    [GlobalSetup]
    public void Setup()
    {
        var logger = new NullLogger<AdvancedMemoryPool<float>>();
        _floatPool = new AdvancedMemoryPool<float>(logger, maxBufferSize: 1024 * 1024);
        
        var byteLogger = new NullLogger<AdvancedMemoryPool<byte>>();
        _bytePool = new AdvancedMemoryPool<byte>(byteLogger, maxBufferSize: 1024 * 1024);
        
        _poolManager = new MemoryPoolManager(NullLoggerFactory.Instance);
    }
    
    [GlobalCleanup]
    public void Cleanup()
    {
        _floatPool?.Dispose();
        _bytePool?.Dispose();
        _poolManager?.Dispose();
    }
    
    [Benchmark]
    public void RentAndReturn_Float()
    {
        var memory = _floatPool.Rent(BufferSize);
        memory.Dispose();
    }
    
    [Benchmark]
    public void RentAndReturn_Byte()
    {
        var memory = _bytePool.Rent(BufferSize);
        memory.Dispose();
    }
    
    [Benchmark]
    public void RentMultiple_NoReturn()
    {
        var memories = new Orleans.GpuBridge.Abstractions.Memory.IGpuMemory<float>[10];
        for (int i = 0; i < 10; i++)
        {
            memories[i] = _floatPool.Rent(BufferSize / 10);
        }
        
        // Return them all
        foreach (var memory in memories)
        {
            memory.Dispose();
        }
    }
    
    [Benchmark]
    public void PoolManager_GetPool()
    {
        var pool = _poolManager.GetPool<double>();
        var memory = pool.Rent(BufferSize);
        memory.Dispose();
    }
    
    [Benchmark]
    public void ConcurrentRentReturn()
    {
        Parallel.For(0, 100, i =>
        {
            var memory = _floatPool.Rent(BufferSize / 100);
            Thread.SpinWait(10);
            memory.Dispose();
        });
    }
    
    [Benchmark]
    public void MemoryPoolWithClear()
    {
        var memory = _bytePool.Rent(BufferSize);
        var span = memory.AsMemory().Span;
        span.Fill(42);
        memory.Dispose(); // Should clear on return
    }
}