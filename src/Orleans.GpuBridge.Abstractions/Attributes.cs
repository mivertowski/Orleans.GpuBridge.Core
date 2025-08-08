using System;
namespace Orleans.GpuBridge.Abstractions;
[AttributeUsage(AttributeTargets.Class|AttributeTargets.Method, AllowMultiple=false)]
public sealed class GpuAcceleratedAttribute:Attribute{ public string KernelId{get;} public GpuAcceleratedAttribute(string id)=>KernelId=id; }
