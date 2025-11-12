```

BenchmarkDotNet v0.14.0, Ubuntu 22.04.5 LTS (Jammy Jellyfish) WSL
Intel Core Ultra 7 165H, 1 CPU, 22 logical and 11 physical cores
.NET SDK 9.0.203
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  DefaultJob : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2


```
| Method                      | Mean         | Error        | StdDev        | Median       | Min          | Max          | Ratio    | RatioSD | Allocated | Alloc Ratio |
|---------------------------- |-------------:|-------------:|--------------:|-------------:|-------------:|-------------:|---------:|--------:|----------:|------------:|
| Now                         |     55.20 ns |     1.476 ns |      4.307 ns |     54.05 ns |     48.03 ns |     67.01 ns |     1.01 |    0.11 |         - |          NA |
| Update                      |     60.05 ns |     3.526 ns |      9.829 ns |     57.09 ns |     43.45 ns |     87.97 ns |     1.09 |    0.20 |         - |          NA |
| CompareTo                   |    112.74 ns |     4.391 ns |     12.166 ns |    109.57 ns |     97.37 ns |    151.43 ns |     2.05 |    0.27 |         - |          NA |
| Now_Batch1000               | 57,794.53 ns | 3,889.861 ns | 11,285.192 ns | 54,974.14 ns | 43,586.81 ns | 88,408.27 ns | 1,053.05 |  219.72 |         - |          NA |
| Update_Batch1000            | 48,661.68 ns | 1,031.954 ns |  2,977.424 ns | 48,095.86 ns | 43,418.87 ns | 55,350.72 ns |   886.64 |   85.56 |         - |          NA |
| Now_AllocationTest          |     51.61 ns |     1.501 ns |      4.133 ns |     51.58 ns |     43.83 ns |     63.51 ns |     0.94 |    0.10 |         - |          NA |
| Now_LogicalCounterIncrement |    507.90 ns |    14.509 ns |     42.323 ns |    499.89 ns |    441.68 ns |    610.02 ns |     9.25 |    1.03 |         - |          NA |
