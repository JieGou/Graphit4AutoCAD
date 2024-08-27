using Newtonsoft.Json.Bson;
using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra; // → CalculateSpectralRadius
using MathNet.Numerics.LinearAlgebra.Double; 

namespace Graphit
{
    internal class Calculations
    {
        public class NodeLevelCalculations
        {
            public static void CalculateNodeDegree(List<Node> nodesList, List<Edge> edgesList)
            {
                foreach (Node node in nodesList)
                {
                    node.degree = edgesList.Count(e => e.start == node || e.end == node);
                }
            }
            
            public static void CalculateNodeEigenvectorCentrality(List<Node> nodesList, List<Edge> edgesList)
            {
                // Initialize centrality score dictionary
                Dictionary<Node, double> centralitiesDict = nodesList.ToDictionary(nd => nd, nd => 1.0);

                double tolerance = 1e-6;
                double dampingFactor = 0.85; // --- Google'nod PageRank damping factor : typically betw. 0.85-0.95
                bool converged = false;

                while (!converged)
                {
                    // Temporary centralitiesDict for this iteration
                    Dictionary<Node, double> tempCentralities = new Dictionary<Node, double>();

                    // Update centralitiesDict for each nod
                    foreach (var node in nodesList)
                    {
                        double sum = 0.0;

                        // Sum of the centralitiesDict of neighbors
                        foreach (var edge in edgesList)
                        {
                            if (edge.start == node)
                            { sum += centralitiesDict[edge.end]; }
                            else if (edge.end == node)
                            { sum += centralitiesDict[edge.start]; }
                        }

                        //tempCentralities[nod] = sum; // --- Freeze/Infinitive loop without Damp.Factor
                        tempCentralities[node] = dampingFactor * sum + (1 - dampingFactor);
                    }

                    // Normalise the centralitiesDict
                    double norm = Math.Sqrt(tempCentralities.Values.Sum(x => x * x)); // --- Euclidean norm

                    foreach (var node in nodesList)
                    {
                        tempCentralities[node] /= norm;
                    }

                    // Check for convergence
                    converged = true;
                    foreach (var node in nodesList)
                    {
                        if (Math.Abs(tempCentralities[node] - centralitiesDict[node]) > tolerance)
                        {
                            converged = false;
                            break;
                        }
                    }

                    //Update centralitiesDict with the values of this iteration
                    centralitiesDict = tempCentralities;
                }

                // Assign the final centralitiesDict to the nodes
                foreach (var node in nodesList)
                {
                    node.centralityEigenvector = centralitiesDict[node] * 100;
                }
            }
            
            public static void CalculateBetweennessCentrality (List<Node> nodesList, List<Edge> edgesList)
            {
                // Initialize betweenness centralitiesDict
                Dictionary<Node, double> centralitiesDict = nodesList.ToDictionary(node => node, node => 0.0);

                foreach (var nod in nodesList)
                {
                    // Step 1: Initialization
                    Stack<Node> stack = new Stack<Node>(); // --- Stack of nodes in order of non-increasing distanceDict from nod
                    Dictionary<Node, List<Node>> predecessors = nodesList.ToDictionary(v => v, v => new List<Node>()); // --- Predecessors in shortest paths from nod to firstNode
                    Dictionary<Node, double> sigma = nodesList.ToDictionary(v => v, v => 0.0);      // --- Number of shortest paths from nod to firstNode
                    Dictionary<Node, double> distance = nodesList.ToDictionary(v => v, v => -1.0);  // --- Distance from nod to firstNode

                    sigma[nod] = 1.0;       // --- Number of shortest paths from nod to nod
                    distance[nod] = 0.0;    // --- Distance from nod to nod

                    Queue<Node> queue = new Queue<Node>(); // --- Queue of nodes in order of increasing distanceDict from nod
                    queue.Enqueue(nod); 

                    // Step 2: BFS to find shortest paths
                    while (queue.Count > 0) // --- While queue is not empty
                    {
                        Node firstNode = queue.Dequeue(); // --- Dequeue the first nod in the queue
                        stack.Push(firstNode); // --- Push firstNode onto the stack

                        // For each neighbor of firstNode
                        foreach (var edg in edgesList)
                        {
                            Node w = null; // --- Neighbor of firstNode

                            // Find the neighbor of firstNode
                            if (edg.start == firstNode)
                            {
                                w = edg.end;
                            }
                            else if (edg.end == firstNode)
                            {
                                w = edg.start;
                            }

                            if (w == null) continue;

                            // w found for the first time?
                            if (distance[w] < 0)
                            {
                                queue.Enqueue(w);
                                distance[w] = distance[firstNode] + 1;
                            }

                            // Shortest path to w via firstNode?
                            if (distance[w] == distance[firstNode] + 1)
                            {
                                sigma[w] += sigma[firstNode];
                                predecessors[w].Add(firstNode);
                            }
                        }
                    }

                    // Step 3: Accumulation
                    Dictionary<Node, double> delta = nodesList.ToDictionary(v => v, v => 0.0); // --- Dependency of nod on firstNode

                    while (stack.Count > 0) // --- While stack is not empty
                    {
                        Node w = stack.Pop(); // --- Pop the top nod from the stack
                        foreach (Node v in predecessors[w]) // --- For each predecessor of w
                        {
                            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]); // --- Accumulate dependency
                        }
                        if (w != nod) // --- Update centrality if w is not nod
                        {
                            centralitiesDict[w] += delta[w]; // --- Accumulate dependency
                        }
                    }
                }

                // Step 4: Normalize the centralitiesDict (for undirected graphs)
                int n = nodesList.Count;

                foreach (var node in nodesList) 
                {
                    centralitiesDict[node] /= (n - 1) * (n - 2) / 2.0; 
                    node.centralityBetweenness = centralitiesDict[node];
                }
            }
            
            public static void CalculateClosenessCentrality (List<Node> nodesList, List<Edge> edgesList)
            {
                foreach (var nod in nodesList)
                {
                    // Step 1: Inıtialization
                    Dictionary<Node, double> distanceDict = nodesList.ToDictionary(v => v, v => double.PositiveInfinity); // --- Distance from nod to firstNode
                    Queue<Node> queue = new Queue<Node>(); // --- Queue of nodes in order of increasing distanceDict from nod

                    distanceDict[nod] = 0; // --- Distance from nod to nod
                    queue.Enqueue(nod); 

                    // Step 2: BFS to find shortest paths
                    while (queue.Count>0)
                    {
                        Node v = queue.Dequeue(); // --- Dequeue the first nod in the queue

                        // For each neighbor of nod n
                        foreach (var edg in edgesList)
                        {
                            Node w = null; // --- Neighbor of nod n
                            if (edg.start == v)
                            {
                                w = edg.end;
                            }
                            else if (edg.end == v)
                            {
                                w = edg.start;
                            }

                            if (w == null)
                            {
                                continue;
                            }

                            // If a shorter path to w is found
                            if (distanceDict[w] == double.PositiveInfinity) // --- 1 is the length of the edg
                            {
                                queue.Enqueue(w);
                                distanceDict[w] = distanceDict[v] + 1;
                            }
                        }
                    }

                    // Step 3: Calculate the closeness centrality
                    double sumDistances = distanceDict.Values.Where(d => d != double.PositiveInfinity).Sum(); 
                    int reachableNodes = distanceDict.Values.Count(d => d != double.PositiveInfinity);

                    // If the nod is isolated or unreachable, set closeness cent. to 0
                    if (reachableNodes > 1)
                    {
                        nod.centralityCloseness = (reachableNodes - 1) / sumDistances;
                    }
                    else
                    {
                        nod.centralityCloseness = 0;
                    }
                }
            }
            
            public static void CalculateClusteringCoefficient (List<Node> nodesList, List<Edge> edgesList)
            {
                // Dictionary to store neighbors for each node
                Dictionary<Node, List<Node>> neighbors = nodesList.ToDictionary(n => n, n => new List<Node>());
            
                // Step 1: Identify all neighbors for each node
                foreach (var edge in edgesList)
                {
                    neighbors[edge.start].Add(edge.end); // --- Add end node to the list of neighbors of start node
                    neighbors[edge.end].Add(edge.start); // --- Add start node to the list of neighbors of end node
                }

                // Step 2: Calculate the clustering coefficient for each node
                foreach (var node in nodesList)
                {
                    List<Node> nodeNeighbors = neighbors[node]; // --- List of neighbors of the node
                    int degree = nodeNeighbors.Count;

                    // If the node has less than 2 neighbors, set clustering coefficient to 0
                    if (degree < 2)
                    {
                        node.clusteringCoefficient = 0;
                        continue;
                    }

                    // Count the number of triangles
                    int triangleCount = 0;

                    for (int i = 0; i < degree; i++)
                    {
                        for (int j = i + 1; j < degree; j++)
                        {
                            if (neighbors[nodeNeighbors[i]].Contains(nodeNeighbors[j]))
                            {
                                triangleCount++;
                            }
                        }
                    }

                    // Calculate the clustering coefficient
                    node.clusteringCoefficient = (2.0 * triangleCount) / (degree * (degree - 1));
                }
            }
        }

        public class GraphLevelCalculations
        {
            public static void CalculateAverageDegree(List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                // Number of nodes in graph
                int nodeCount = nodesList.Count;

                // Number of edges
                int edgeCount = edgesList.Count;

                // Calculate the average key
                double averageDegree = 0.0;

                if (nodeCount > 0)
                {
                    averageDegree = (2.0 * edgeCount) / nodeCount;
                }

                // Assign the average key to the graph
                graph.averageDegree = averageDegree;
            }
           
            public static void CalculateGraphDensity (List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                int n = nodesList.Count;
                int m = edgesList.Count;

                // Calculate the density
                double density = (2.0 * m) / (n * (n - 1));

                // Assign the density to the graph
                graph.density = density;
            }

            //////////////////////////////////////////////////////////////////////////////////////////

            public static void CalculateGraphDiameter (List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                // Initialize the diameter
                int diameter = 0;

                // For each node, perform BFS to find the longest shortest path from that node
                foreach (var sourceNode in nodesList)
                {
                    // Use BFS to find the shortest paths from the source node
                    Dictionary<Node, int> shortestPaths = BFSShortestPaths(sourceNode, nodesList, edgesList);

                    // Find the maximum distance from sourceNode to any other node
                    int maxDistance = shortestPaths.Values.Max();

                    // Update the diameter if the current maxDistance is larger
                    if (maxDistance > diameter)
                    {
                        diameter = maxDistance;
                    }
                }

                // Assign the diameter to the graph
                graph.diameter = diameter;
            }

            // Helper method → Calculate Graph Diameter
            private static Dictionary<Node, int> BFSShortestPaths(Node startNode, List<Node> nodesList, List<Edge> edgesList)
            {
                // Initialize the distance dictionary
                Dictionary<Node, int> distanceDict = nodesList.ToDictionary(n => n, n => int.MaxValue);

                // Initialize the queue for BFS
                Queue<Node> queue = new Queue<Node>();

                // Start node has distance 0
                distanceDict[startNode] = 0;
                queue.Enqueue(startNode);

                // Perform BFS
                while (queue.Count > 0)
                {
                    Node currnetNode = queue.Dequeue();

                    // Explore neighbors
                    foreach (var edge in edgesList)
                    {
                        Node neighbor = null;

                        if (edge.start == currnetNode)
                        {
                            neighbor = edge.end;
                        }
                        else if (edge.end == currnetNode)
                        {
                            neighbor = edge.start;
                        }

                        if (neighbor == null)
                        {
                            continue;
                        }

                        // If the distance to the neighbor can be minimized, update it
                        if (distanceDict[neighbor] > distanceDict[currnetNode] + 1)
                        {
                            distanceDict[neighbor] = distanceDict[currnetNode] + 1; // --- 1 is the length of the edge
                            queue.Enqueue(neighbor);
                        }
                    }
                }

                return distanceDict;
            }

            //////////////////////////////////////////////////////////////////////////////////////////
            
            public static void CalculateGraphKernelDegree (List<Node> nodesList, Graph graph)
            {
                // Initialize the kernel dictionary to store the key of each node
                Dictionary<int, int> kernelDegreeDict = new Dictionary<int, int>();

                // Initialize node key dictionary
                Dictionary<Node, int> degreeDict = nodesList.ToDictionary(n => n, n => 0);

                // Step.1 : Get the key of each node from nodesList objects value node.key
                foreach (var node in nodesList)
                {
                    degreeDict[node] = node.degree; 
                }

                foreach (var degree in degreeDict.Values)
                {
                    if(kernelDegreeDict.ContainsKey(degree))
                    {
                        kernelDegreeDict[degree]++;
                    }
                    else
                    {
                        kernelDegreeDict[degree] = 1;
                    }
                }

                // Step.2 : Assign the kernel key dictionary to the graph
                graph.kernelDegree = kernelDegreeDict;
            }

            public static void CalculateKernelDegreeDistribution(List<Node> nodesList, Graph graph)
            {
                // Step 1: Get the key kernel information from the parameter of key kernel of the graph
                Dictionary<int, int> kernelDegreeDict = graph.kernelDegree;

                // Step 2: Convert "kernelDegreeDict" to a <int, double> dictionary
                Dictionary<int, double> degreeDistributionDict = kernelDegreeDict.ToDictionary(kv => kv.Key, kv => (double)kv.Value);

                int totalNodes = nodesList.Count;

                // Step 3 : Normalize the values of "degreeDistributionDict" with "totalNodes"
                foreach (var key in degreeDistributionDict.Keys.ToList())
                {
                    degreeDistributionDict[key] /= totalNodes;
                }

                // Step 4: Assign the key distribution dictionary to the graph
                graph.kernelDegreeDistribution = degreeDistributionDict;
            }

            //////////////////////////////////////////////////////////////////////////////////////////

            public static void CalculateConnectedComponents(List<Node> nodesList, List<Edge> edgesList, Graph graph) 
            {
                // Step 1: Initialize the dictionary to keep track of visited nodes and
                // the counter for connected components
                HashSet<Node> visited = new HashSet<Node>();
                int connectedComponents = 0;

                // Step 2: Iterate over all nodes in the graph
                foreach (var node in nodesList)
                {
                    if(!visited.Contains(node))
                    {
                        connectedComponents++;
                        // Perform DFS from the current node
                        DFS(node, visited, edgesList);
                    }
                }

                // Step 3: Assign the number of connected components to the graph
                graph.connectedComponents = connectedComponents;

            }

            // Helper method → Calculate Connected Components
            private static void DFS(Node node, HashSet<Node> visited, List<Edge> edgesList)
            {
                // Mark the node as visited
                visited.Add(node);

                // Explore all neighbors (connected nodes)
                foreach (var edge in edgesList)
                {
                    // Check both ends of the edge to find neighbors
                    Node neighbor = null;

                    if (edge.start == node)
                    {
                        neighbor = edge.end;
                    }
                    else if (edge.end == node)
                    {
                        neighbor = edge.start;
                    }

                    if (neighbor == null)
                    {
                        continue;
                    }

                    if (!visited.Contains(neighbor) && neighbor != null)
                    {
                        // Recursive visit to neighbor
                        DFS(neighbor, visited, edgesList);
                    }
                }
            }

            //////////////////////////////////////////////////////////////////////////////////////////
        
            public static void CalculateAveragePathLength(List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                // Step 1: Initialize the sum of shortest path lengths and the counter of shortest paths
                double sumPathLengths = 0.0;
                int pathCount = 0;

                // Step 2: Iterate over all pairs of nodes
                foreach (var startNode in nodesList)
                {
                    // Use BFS to find the shortest paths from the start node
                    Dictionary<Node, int> shortestPaths = BFSShortestPaths(startNode, nodesList, edgesList);

                    // Iterate over all other nodes
                    foreach (var endNode in nodesList)
                    {
                        // Skip the same node
                        if (startNode == endNode)
                        {
                            continue;
                        }

                        // Get the shortest path length from the dictionary
                        int pathLength = shortestPaths[endNode];

                        // If the path length is not infinite, add it to the sum
                        if (pathLength != int.MaxValue)
                        {
                            sumPathLengths += pathLength;
                            pathCount++;
                        }
                    }
                }

                // Step 3: Calculate the average path length
                double averagePathLength = sumPathLengths / pathCount;

                // Step 4: Assign the average path length to the graph
                graph.averagePathLength = averagePathLength;
            }

            public static void CalculateAssortivityCoefficient(List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                // Step 1: Initialize the variables
                double sumDegreeProduct = 0.0;
                double sumSquaredDegreeSum = 0.0;
                double sumDegreeSum = 0.0;

                // Step 2: Calculate the sums
                foreach (var edge in edgesList)
                {
                    sumDegreeProduct += edge.start.degree * edge.end.degree;
                    sumSquaredDegreeSum += Math.Pow(edge.start.degree, 2) + Math.Pow(edge.end.degree, 2);
                    sumDegreeSum += edge.start.degree + edge.end.degree;
                }

                // Step 3: Calculate the assortativity coefficient according to the formula of Newman
                double assortativityCoefficient = (sumDegreeProduct / edgesList.Count - Math.Pow(sumDegreeSum / (2 * edgesList.Count), 2)) /
                                                  (sumSquaredDegreeSum / (2 * edgesList.Count) - Math.Pow(sumDegreeSum / (2 * edgesList.Count), 2));

                // Step 4: Assign the assortativity coefficient to the graph
                graph.assortivityCoefficient = assortativityCoefficient;
            }

            public static void CalculateTransitivity(List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                // Step 1: Initialize the variables
                int triangles = 0;
                int triplets = 0;

                // Step 2: Calculate the number of triangles and triplets
                foreach (var node in nodesList)
                {
                    // Get the neighbors of the node
                    List<Node> neighbors = edgesList.Where(e => e.start == node || e.end == node)
                        .Select(e => e.start == node ? e.end : e.start).ToList();

                    // Count the number of triangles and triplets
                    for (int i = 0; i < neighbors.Count; i++)
                    {
                        for (int j = i + 1; j < neighbors.Count; j++)
                        {
                            if (edgesList.Any(e => (e.start == neighbors[i] && e.end == neighbors[j]) ||
                                                                               (e.start == neighbors[j] && e.end == neighbors[i])))
                            {
                                triangles++;
                            }
                            triplets++;
                        }
                    }
                }

                // Step 3: Calculate the transitivity
                double transitivity = 0.0;

                if (triplets > 0)
                {
                    transitivity = 3.0 * triangles / triplets;
                }

                // Step 4: Assign the transitivity to the graph
                graph.transitivity = transitivity;
            }
        
            public static void CalculateSpectralRadius(List<Node> nodesList, List<Edge> edgesList, Graph graph )
            {
                // Step 1: Initialize variables
                int n = nodesList.Count;
                Matrix<double> adjacencyMatrix = Matrix<double>.Build.Dense(n, n);
                Dictionary<Node, int> nodeIndexDict = new Dictionary<Node, int>();

                // Step 2: Construct the "nodeIndexDict" dictioanry
                for (int i = 0; i<n; i++)
                {
                    nodeIndexDict[nodesList[i]] = i;
                }

                // Step 3: Construct the adjacency matrix (for an undirected graph)
                foreach (var edge in edgesList)
                {
                    int i = nodeIndexDict[edge.start];
                    int j = nodeIndexDict[edge.end];

                    adjacencyMatrix[i, j] = 1;
                    adjacencyMatrix[j, i] = 1;
                }

                // Step 4: Calculate the eigenvalues of the adjacency matrix
                var eigenvalues = adjacencyMatrix.Evd().EigenValues;

                // Step 5: Find the largest eigenvalue (which is spectral radius)
                double spectralRadius = eigenvalues.Real().Max();

                // Step 6: Assign the spectral radius to the graph
                graph.spectralRadius = spectralRadius;
            }
        
            public static void CalculateAlgebraicConnectivity(List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                // Step 1: Initialize variables
                int n = nodesList.Count;
                Matrix<double> laplacianMatrix = Matrix<double>.Build.Dense(n, n);
                Dictionary<Node, int> nodeIndexDict = new Dictionary<Node, int>();

                // Step 2: Construct the "nodeIndexDict" dictionary
                for (int i = 0; i < n; i++)
                {
                    nodeIndexDict[nodesList[i]] = i;
                }

                // Step 3: Construct the Laplacian matrix
                foreach (var edge in edgesList)
                {
                    int i = nodeIndexDict[edge.start];
                    int j = nodeIndexDict[edge.end];

                    laplacianMatrix[i, j] = -1;
                    laplacianMatrix[j, i] = -1;
                    laplacianMatrix[i, i] += 1;
                    laplacianMatrix[j, j] += 1;
                }

                // Step 4: Calculate the eigenvalues of the Laplacian matrix
                var eigenvalues = laplacianMatrix.Evd().EigenValues;

                // Step 5: Find the second smallest eigenvalue (which is algebraic connectivity)
                double[] realEigenvalues = eigenvalues.Real().ToArray();
                Array.Sort(realEigenvalues);
                double algebraicConnectivity = realEigenvalues[1];

                // Step 6: Assign the algebraic connectivity to the graph
                graph.algebraicConnectivity = algebraicConnectivity;
            }

            public static void CalculateEigenvalueDistribution(List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                int n = nodesList.Count;

                // Step 1: Create the adjacency matrix
                Matrix<double> adjacencyMatrix = Matrix<double>.Build.Dense(n, n);
                Dictionary<Node, int> nodeIndex = new Dictionary<Node, int>();

                // Step 2: Construct the adjacency matrix
                for (int i = 0; i < n; i++)
                {
                    nodeIndex[nodesList[i]] = i;
                }

                foreach (var edge in edgesList)
                {
                    int i = nodeIndex[edge.start];
                    int j = nodeIndex[edge.end];

                    adjacencyMatrix[i, j] = 1;
                    adjacencyMatrix[j, i] = 1;
                }

                // Step 3: Calculate the eigenvalues of the adjacency matrix
                var eigenvalues = adjacencyMatrix.Evd().EigenValues;

                // Step 4: Convert the eigenvalues to a dictionary
                Dictionary<double, int> eigenvalueDistribution = new Dictionary<double, int>();

                // Step 5: Count the number of eigenvalues in each bin
                foreach (var eigenvalue in eigenvalues.Real())
                {
                    double bin = Math.Round(eigenvalue, 2);

                    if (eigenvalueDistribution.ContainsKey(bin))
                    {
                        eigenvalueDistribution[bin]++;
                    }
                    else
                    {
                        eigenvalueDistribution[bin] = 1;
                    }
                }

                // Step 6: Assign the eigenvalue distribution to the graph
                graph.eigenvalueDistribution = eigenvalueDistribution;
            }
        
            public static void CalculateModularity(List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                int n = nodesList.Count;
                int m = edgesList.Count;

                // Step 1: Calculate the total degree of the graph
                double totalDegree = nodesList.Sum(nod => nod.degree);

                // Step 2: Calculate the modularity matrix
                Matrix<double> modularityMatrix = Matrix<double>.Build.Dense(n, n);

                foreach (var edge in edgesList)
                {
                    int i = nodesList.IndexOf(edge.start);
                    int j = nodesList.IndexOf(edge.end);

                    modularityMatrix[i, j] = 1;
                    modularityMatrix[j, i] = 1;
                }

                // Step 3: Calculate the modularity
                double modularity = 0.0;

                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double ki = nodesList[i].degree;
                        double kj = nodesList[j].degree;

                        modularityMatrix[i, j] -= (ki * kj) / (2 * m);
                    }
                }

                // Step 4: Calculate the eigenvalues of the modularity matrix
                var eigenvalues = modularityMatrix.Evd().EigenValues;

                // Step 5: Calculate the modularity
                foreach (var eigenvalue in eigenvalues.Real())
                {
                    modularity += eigenvalue;
                }

                // Step 6: Normalize the modularity
                modularity /= (2 * m);

                // Step 7: Assign the modularity to the graph
                graph.modularity = modularity;
            }
        }
    }
}
