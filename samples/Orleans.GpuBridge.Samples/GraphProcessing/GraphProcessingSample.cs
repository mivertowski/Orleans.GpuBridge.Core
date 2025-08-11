using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions;
using Spectre.Console;

namespace Orleans.GpuBridge.Samples.GraphProcessing;

public class GraphProcessingSample
{
    private readonly IServiceProvider _services;
    private readonly IGpuBridge _gpuBridge;
    
    public GraphProcessingSample(IServiceProvider services)
    {
        _services = services;
        _gpuBridge = services.GetRequiredService<IGpuBridge>();
    }
    
    public async Task RunAsync(int nodeCount, int edgeCount, string algorithm)
    {
        AnsiConsole.MarkupLine($"[bold]Graph Processing Sample[/]");
        AnsiConsole.MarkupLine($"Nodes: [cyan]{nodeCount:N0}[/]");
        AnsiConsole.MarkupLine($"Edges: [cyan]{edgeCount:N0}[/]");
        AnsiConsole.MarkupLine($"Algorithm: [cyan]{algorithm}[/]");
        AnsiConsole.WriteLine();
        
        var graph = GenerateRandomGraph(nodeCount, edgeCount);
        
        switch (algorithm.ToLower())
        {
            case "pagerank":
                await RunPageRankAsync(graph);
                break;
            case "shortest":
                await RunShortestPathAsync(graph);
                break;
            case "traversal":
                await RunBreadthFirstTraversalAsync(graph);
                break;
            case "all":
            default:
                await RunPageRankAsync(graph);
                await RunShortestPathAsync(graph);
                await RunBreadthFirstTraversalAsync(graph);
                break;
        }
    }
    
    public async Task RunInteractiveAsync()
    {
        var nodes = AnsiConsole.Ask<int>("Enter number of nodes:", 10000);
        var edges = AnsiConsole.Ask<int>("Enter number of edges:", 50000);
        var algorithm = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Select algorithm:")
                .AddChoices("PageRank", "Shortest Path", "BFS Traversal", "All"));
        
        await RunAsync(nodes, edges, algorithm);
    }
    
    private async Task RunPageRankAsync(GraphData graph)
    {
        AnsiConsole.MarkupLine("[underline]PageRank Algorithm[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var parameters = new PageRankParameters
        {
            Iterations = 20,
            DampingFactor = 0.85f,
            Tolerance = 0.0001f
        };
        
        var sw = Stopwatch.StartNew();
        var result = await _gpuBridge.ExecuteAsync<(GraphData, PageRankParameters), PageRankResult>(
            "graph/pagerank",
            (graph, parameters));
        sw.Stop();
        
        table.AddRow("Processing Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Iterations", $"{parameters.Iterations}");
        table.AddRow("Nodes Processed", $"{graph.NodeCount:N0}");
        table.AddRow("Edges Processed", $"{graph.EdgeCount:N0}");
        table.AddRow("Throughput", $"{graph.EdgeCount / sw.Elapsed.TotalSeconds:N0} edges/sec");
        
        // Show top ranked nodes
        if (result.Rankings != null && result.Rankings.Length > 0)
        {
            table.AddRow("Top Node Score", $"{result.Rankings[0]:F6}");
        }
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task RunShortestPathAsync(GraphData graph)
    {
        AnsiConsole.MarkupLine("[underline]Shortest Path (Dijkstra)[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var sourceNode = 0;
        var targetNode = graph.NodeCount - 1;
        
        var sw = Stopwatch.StartNew();
        var result = await _gpuBridge.ExecuteAsync<(GraphData, int, int), ShortestPathResult>(
            "graph/shortest",
            (graph, sourceNode, targetNode));
        sw.Stop();
        
        table.AddRow("Processing Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Source Node", $"{sourceNode}");
        table.AddRow("Target Node", $"{targetNode}");
        table.AddRow("Path Length", result.Distance < float.MaxValue ? $"{result.Distance:F2}" : "No path");
        table.AddRow("Nodes Explored", $"{result.NodesExplored:N0}");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task RunBreadthFirstTraversalAsync(GraphData graph)
    {
        AnsiConsole.MarkupLine("[underline]Breadth-First Traversal[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var startNode = 0;
        
        var sw = Stopwatch.StartNew();
        var result = await _gpuBridge.ExecuteAsync<(GraphData, int), TraversalResult>(
            "graph/bfs",
            (graph, startNode));
        sw.Stop();
        
        table.AddRow("Processing Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Start Node", $"{startNode}");
        table.AddRow("Nodes Visited", $"{result.VisitedCount:N0}");
        table.AddRow("Max Depth", $"{result.MaxDepth}");
        table.AddRow("Throughput", $"{result.VisitedCount / sw.Elapsed.TotalSeconds:N0} nodes/sec");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private GraphData GenerateRandomGraph(int nodeCount, int edgeCount)
    {
        var random = new Random(42);
        var edges = new List<(int, int, float)>();
        var edgeSet = new HashSet<(int, int)>();
        
        // Generate random edges
        while (edges.Count < edgeCount)
        {
            var from = random.Next(nodeCount);
            var to = random.Next(nodeCount);
            
            if (from != to && edgeSet.Add((from, to)))
            {
                var weight = (float)(random.NextDouble() * 10 + 1);
                edges.Add((from, to, weight));
            }
        }
        
        // Convert to CSR format for GPU processing
        var (rowOffsets, colIndices, values) = ConvertToCSR(nodeCount, edges);
        
        return new GraphData
        {
            NodeCount = nodeCount,
            EdgeCount = edges.Count,
            RowOffsets = rowOffsets,
            ColumnIndices = colIndices,
            Values = values
        };
    }
    
    private (int[], int[], float[]) ConvertToCSR(int nodeCount, List<(int from, int to, float weight)> edges)
    {
        // Sort edges by source node
        edges.Sort((a, b) => a.from.CompareTo(b.from));
        
        var rowOffsets = new int[nodeCount + 1];
        var colIndices = new int[edges.Count];
        var values = new float[edges.Count];
        
        int currentNode = 0;
        for (int i = 0; i < edges.Count; i++)
        {
            var (from, to, weight) = edges[i];
            
            while (currentNode <= from)
            {
                rowOffsets[currentNode++] = i;
            }
            
            colIndices[i] = to;
            values[i] = weight;
        }
        
        while (currentNode <= nodeCount)
        {
            rowOffsets[currentNode++] = edges.Count;
        }
        
        return (rowOffsets, colIndices, values);
    }
}

public class GraphData
{
    public int NodeCount { get; set; }
    public int EdgeCount { get; set; }
    public int[] RowOffsets { get; set; } = Array.Empty<int>();
    public int[] ColumnIndices { get; set; } = Array.Empty<int>();
    public float[] Values { get; set; } = Array.Empty<float>();
}

public class PageRankParameters
{
    public int Iterations { get; set; }
    public float DampingFactor { get; set; }
    public float Tolerance { get; set; }
}

public class PageRankResult
{
    public float[] Rankings { get; set; } = Array.Empty<float>();
    public int ConvergedAtIteration { get; set; }
}

public class ShortestPathResult
{
    public float Distance { get; set; }
    public int[] Path { get; set; } = Array.Empty<int>();
    public int NodesExplored { get; set; }
}

public class TraversalResult
{
    public int VisitedCount { get; set; }
    public int MaxDepth { get; set; }
    public int[] VisitOrder { get; set; } = Array.Empty<int>();
}