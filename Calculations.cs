using Newtonsoft.Json.Bson;
using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
            public static void CalculateGraphDensity (List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                int n = nodesList.Count;
                int m = edgesList.Count;

                // Calculate the density
                double density = (2.0 * m) / (n * (n - 1));

                // Assign the density to the graph
                graph.density = density;
            }
            public static void CalculateGraphKernelDegree (List<Node> nodesList, List<Edge> edgesList, Graph graph)
            {
                // Initialize the kernel dictionary to store the degree of each node
                Dictionary<int, int> kernelDegreeDict = new Dictionary<int, int>();

                // Initialize node degree dictionary
                Dictionary<Node, int> degreeDict = nodesList.ToDictionary(n => n, n => 0);

                // Step.1 Get the degree of each node from nodesList objects value node.degree
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

                // Step.2 Assign the kernel degree dictionary to the graph
                graph.kernelDegree = kernelDegreeDict;
            }
        
        }
    }
}
