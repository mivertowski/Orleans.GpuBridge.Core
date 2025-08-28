// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Text.Json;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class SerializationBenchmarks
{
    private KernelInfo _kernelInfo = null!;
    private KernelHandle _kernelHandle = null!;
    private GpuExecutionHints _executionHints = null!;
    private string _jsonKernelInfo = null!;
    private byte[] _binaryKernelInfo = null!;
    
    [GlobalSetup]
    public void Setup()
    {
        _kernelInfo = new KernelInfo(
            new KernelId("test-kernel"),
            "Test Kernel",
            typeof(float[]),
            typeof(float),
            true,
            1024);
        
        _kernelHandle = KernelHandle.Create();
        
        _executionHints = new GpuExecutionHints
        {
            PreferredDeviceIndex = 0,
            MaxBatchSize = 1000,
            TimeoutMs = 5000,
            Priority = ExecutionPriority.Normal
        };
        
        _jsonKernelInfo = JsonSerializer.Serialize(_kernelInfo);
        _binaryKernelInfo = JsonSerializer.SerializeToUtf8Bytes(_kernelInfo);
    }
    
    [Benchmark]
    public string SerializeKernelInfo_Json()
    {
        return JsonSerializer.Serialize(_kernelInfo);
    }
    
    [Benchmark]
    public byte[] SerializeKernelInfo_Binary()
    {
        return JsonSerializer.SerializeToUtf8Bytes(_kernelInfo);
    }
    
    [Benchmark]
    public KernelInfo? DeserializeKernelInfo_Json()
    {
        return JsonSerializer.Deserialize<KernelInfo>(_jsonKernelInfo);
    }
    
    [Benchmark]
    public KernelInfo? DeserializeKernelInfo_Binary()
    {
        return JsonSerializer.Deserialize<KernelInfo>(_binaryKernelInfo);
    }
    
    [Benchmark]
    public string SerializeExecutionHints()
    {
        return JsonSerializer.Serialize(_executionHints);
    }
    
    [Benchmark]
    public int KernelHandleOperations()
    {
        var handle = KernelHandle.Create();
        var id = handle.Id;
        var timestamp = handle.CreatedAt;
        return id.GetHashCode() + timestamp.GetHashCode();
    }
    
    [Benchmark]
    public void BatchSerialization()
    {
        var batch = new float[100][];
        for (int i = 0; i < 100; i++)
        {
            batch[i] = new float[1024];
        }
        
        var json = JsonSerializer.Serialize(batch);
        var deserialized = JsonSerializer.Deserialize<float[][]>(json);
    }
}