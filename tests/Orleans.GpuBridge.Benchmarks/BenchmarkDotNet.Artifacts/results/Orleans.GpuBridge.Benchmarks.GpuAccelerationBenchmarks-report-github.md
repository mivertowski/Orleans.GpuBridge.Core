```

BenchmarkDotNet v0.14.0, Ubuntu 22.04.5 LTS (Jammy Jellyfish) WSL
Intel Core Ultra 7 165H, 1 CPU, 22 logical and 11 physical cores
.NET SDK 9.0.203
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  DefaultJob : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2


```
| Method                    | Mean            | Error         | StdDev          | Median          | Min             | Max             | Ratio     | RatioSD  | Allocated | Alloc Ratio |
|-------------------------- |----------------:|--------------:|----------------:|----------------:|----------------:|----------------:|----------:|---------:|----------:|------------:|
| VectorAdd_Cpu_Scalar_1K   |        874.7 ns |      47.91 ns |       140.51 ns |        881.5 ns |        510.2 ns |      1,168.0 ns |      1.03 |     0.25 |         - |          NA |
| VectorAdd_Cpu_Scalar_100K |    101,570.6 ns |   9,083.14 ns |    26,781.85 ns |     93,786.0 ns |     59,341.4 ns |    172,158.6 ns |    119.59 |    38.94 |         - |          NA |
| VectorAdd_Cpu_Scalar_1M   |  1,091,693.8 ns |  40,254.65 ns |   117,424.71 ns |  1,057,243.5 ns |    849,132.0 ns |  1,405,131.5 ns |  1,285.40 |   277.61 |       1 B |          NA |
| VectorAdd_Cpu_Simd_1K     |        345.3 ns |      15.63 ns |        45.84 ns |        340.1 ns |        278.3 ns |        468.5 ns |      0.41 |     0.09 |         - |          NA |
| VectorAdd_Cpu_Simd_100K   |     29,747.4 ns |   2,637.12 ns |     7,775.62 ns |     26,550.6 ns |     14,255.3 ns |     49,444.1 ns |     35.03 |    11.34 |         - |          NA |
| VectorAdd_Cpu_Simd_1M     |    642,494.6 ns |  34,166.73 ns |    98,578.85 ns |    615,264.3 ns |    405,384.7 ns |    920,010.3 ns |    756.50 |   183.60 |       1 B |          NA |
| VectorAdd_Gpu_1K          |  5,222,785.6 ns | 140,377.90 ns |   395,938.39 ns |  5,168,750.2 ns |  3,905,325.7 ns |  6,260,299.7 ns |  6,149.49 | 1,240.26 |    3642 B |          NA |
| VectorAdd_Gpu_100K        |  5,985,480.3 ns | 166,878.46 ns |   470,683.69 ns |  5,908,309.0 ns |  4,994,533.3 ns |  7,290,417.7 ns |  7,047.52 | 1,429.17 |    3642 B |          NA |
| VectorAdd_Gpu_1M          | 28,458,836.3 ns | 982,731.09 ns | 2,851,081.28 ns | 28,198,986.1 ns | 23,143,438.9 ns | 36,249,357.7 ns | 33,508.45 | 7,114.02 |    3768 B |          NA |
