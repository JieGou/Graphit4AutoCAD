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
            // Calculate nd degree
            public static void CalculateNodeDegree(List<Node> nodesList, List<Edge> edgesList)
            {
                foreach (Node node in nodesList)
                {
                    node.degree = edgesList.Count(e => e.start == node || e.end == node);
                }
            }

            // Calculate nd eigenvector centrality
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
                    Stack<Node> stack = new Stack<Node>(); // --- Stack of nodes in order of non-increasing distance from nod
                    Dictionary<Node, List<Node>> predecessors = nodesList.ToDictionary(v => v, v => new List<Node>()); // --- Predecessors in shortest paths from nod to firstNode
                    Dictionary<Node, double> sigma = nodesList.ToDictionary(v => v, v => 0.0);      // --- Number of shortest paths from nod to firstNode
                    Dictionary<Node, double> distance = nodesList.ToDictionary(v => v, v => -1.0);  // --- Distance from nod to firstNode

                    sigma[nod] = 1.0;       // --- Number of shortest paths from nod to nod
                    distance[nod] = 0.0;    // --- Distance from nod to nod

                    Queue<Node> queue = new Queue<Node>(); // --- Queue of nodes in order of increasing distance from nod
                    queue.Enqueue(nod); 

                    // Step 2: BFS to find shortest paths
                    while (queue.Count > 0) // --- While queue is not empty
                    {
                        Node firstNode = queue.Dequeue(); // --- Dequeue the first node in the queue
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
                        Node w = stack.Pop(); // --- Pop the top node from the stack
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
        }
        
        

    }
}
