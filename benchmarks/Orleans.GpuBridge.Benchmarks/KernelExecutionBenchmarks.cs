// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class KernelExecutionBenchmarks
{
    private CpuVectorAddKernel _vectorAddKernel = null!;
    private CpuPassthroughKernel<float[], float[]> _passthroughKernel = null!;
    private float[][] _smallBatch = null!;
    private float[][] _mediumBatch = null!;
    private float[][] _largeBatch = null!;
    
    [Params(10, 100, 1000)]
    public int BatchSize { get; set; }
    
    [GlobalSetup]
    public void Setup()
    {
        _vectorAddKernel = new CpuVectorAddKernel();
        _passthroughKernel = new CpuPassthroughKernel<float[], float[]>();
        
        // Create batches
        _smallBatch = CreateBatch(10, 1024);
        _mediumBatch = CreateBatch(100, 1024);
        _largeBatch = CreateBatch(1000, 1024);
    }
    
    private float[][] CreateBatch(int size, int vectorLength)
    {
        var batch = new float[size][];
        for (int i = 0; i < size; i++)
        {
            batch[i] = new float[vectorLength];
            Random.Shared.NextSingle();
            for (int j = 0; j < vectorLength; j++)
            {
                batch[i][j] = Random.Shared.NextSingle();
            }
        }
        return batch;
    }
    
    [Benchmark]
    public async Task VectorAdd_SmallBatch()
    {
        var handle = await _vectorAddKernel.SubmitBatchAsync(_smallBatch);
        var results = new List<float>();
        await foreach (var result in _vectorAddKernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }
    }
    
    [Benchmark]
    public async Task VectorAdd_MediumBatch()
    {
        var handle = await _vectorAddKernel.SubmitBatchAsync(_mediumBatch);
        var results = new List<float>();
        await foreach (var result in _vectorAddKernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }
    }
    
    [Benchmark]
    public async Task Passthrough_Batch()
    {
        var batch = BatchSize switch
        {
            10 => _smallBatch,
            100 => _mediumBatch,
            _ => _largeBatch
        };
        
        var handle = await _passthroughKernel.SubmitBatchAsync(batch);
        var results = new List<float[]>();
        await foreach (var result in _passthroughKernel.ReadResultsAsync(handle))
        {
            results.Add(result);
        }
    }
    
    [Benchmark]
    public float CpuVectorAdd_Direct()
    {
        var a = _smallBatch[0];
        var b = _smallBatch.Length > 1 ? _smallBatch[1] : a;
        return CpuVectorAddKernel.Execute(a, b);
    }
    
    [Benchmark]
    public async Task<KernelInfo> GetKernelInfo()
    {
        return await _vectorAddKernel.GetInfoAsync();
    }
}