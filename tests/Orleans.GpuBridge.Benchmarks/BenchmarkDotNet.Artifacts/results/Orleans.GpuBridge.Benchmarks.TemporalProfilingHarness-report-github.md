```

BenchmarkDotNet v0.14.0, Ubuntu 22.04.5 LTS (Jammy Jellyfish) WSL
Intel Core Ultra 7 165H, 1 CPU, 22 logical and 11 physical cores
.NET SDK 9.0.203
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  Job-CXGQGO : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  Job-WJIJIW : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2

IterationCount=100  WarmupCount=10  

```
| Method                                | Job        | InvocationCount | UnrollFactor | Mean           | Error         | StdDev        | Median         | Gen0   | Gen1   | Allocated |
|-------------------------------------- |----------- |---------------- |------------- |---------------:|--------------:|--------------:|---------------:|-------:|-------:|----------:|
| &#39;Graph: Add edge&#39;                     | Job-CXGQGO | 1               | 1            | 10,062.9565 ns |   603.8077 ns | 1,703.0503 ns |  9,738.5000 ns |      - |      - |    1744 B |
| &#39;HLC: Generate timestamp&#39;             | Job-WJIJIW | Default         | 16           |     51.1471 ns |     0.9706 ns |     2.8158 ns |     50.4210 ns |      - |      - |         - |
| &#39;HLC: Update with received timestamp&#39; | Job-WJIJIW | Default         | 16           |     81.1575 ns |     2.2703 ns |     6.5503 ns |     80.5689 ns |      - |      - |         - |
| &#39;HLC: Compare timestamps&#39;             | Job-WJIJIW | Default         | 16           |      0.1409 ns |     0.0511 ns |     0.1424 ns |      0.1220 ns |      - |      - |         - |
| &#39;Graph: Query time range&#39;             | Job-WJIJIW | Default         | 16           |    541.8434 ns |    20.6602 ns |    56.9041 ns |    525.2010 ns | 0.0238 |      - |     304 B |
| &#39;Graph: Find temporal paths&#39;          | Job-WJIJIW | Default         | 16           |  6,005.7806 ns |   193.3014 ns |   538.8472 ns |  5,900.9102 ns | 0.6409 | 0.0076 |    8104 B |
| &#39;Graph: Get reachable nodes&#39;          | Job-WJIJIW | Default         | 16           | 62,675.1198 ns | 1,769.3509 ns | 4,961.4499 ns | 62,074.1489 ns | 2.9907 | 0.0610 |   38032 B |
| &#39;Graph: Get statistics&#39;               | Job-WJIJIW | Default         | 16           |      0.0159 ns |     0.0147 ns |     0.0433 ns |      0.0000 ns |      - |      - |         - |
| &#39;Combined: HLC + Graph query&#39;         | Job-WJIJIW | Default         | 16           |    388.9052 ns |     9.4974 ns |    27.0967 ns |    384.8867 ns | 0.0238 |      - |     304 B |
