// AutoCad to graph conversion tool
// Extracts Autocad objects to a graph database and calculates graph and node level statistics
// Serializes the graph object to JSON and writes it to a file

using Autodesk.AutoCAD.Runtime; 
// using Autodesk.AutoCAD.ApplicationServices; 
using Autodesk.AutoCAD.DatabaseServices; 
using Autodesk.AutoCAD.Geometry;

using Newtonsoft.Json;

using System.Collections.Generic;
using System.IO;
using System.Linq;

using static Graphit.Calculations;

namespace Graphit
{
    public class Graphit : BaseCommand
    {
        // Extract Autocad objects to graph database

        [CommandMethod("EXTRACTTOGRAPH")]
        public void ExtractToGraph()
        {
            // Start a transaction
            using (Transaction tr = db.TransactionManager.StartTransaction())
            {
                try
                {
                    // Open the Block table & Block table record for read
                    BlockTable bt = tr.GetObject(db.BlockTableId, OpenMode.ForRead) as BlockTable;
                    BlockTableRecord btr = tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead) as BlockTableRecord;

                    List<Node> nodesList = new List<Node>();
                    List<Edge> edgesList = new List<Edge>();

                    // debugging
                    ed.WriteMessage("\n1-Nodes & Edges counting...");

                    // Iterate through all the objects in Model space
                    foreach (ObjectId objId in btr)
                    {
                        Entity ent = tr.GetObject(objId, OpenMode.ForRead) as Entity;

                        if (ent is Line line)
                        {
                            string lineId = line.ObjectId.ToString();
                            string lineStartId = lineId + "s";
                            string lineEndId = lineId + "e";

                            // Check if start node already exists
                            Node startNode = FindOrCreateNode(line.StartPoint, "geometry", lineStartId, "line", nodesList);

                            // Check if end node already exists
                            Node endNode = FindOrCreateNode(line.EndPoint, "geometry", lineEndId, "line", nodesList);

                            Edge edge = new Edge(startNode, endNode, "geometry", lineId, "connection");
                            edgesList.Add(edge);
                        }

                        if (ent is Circle circle)
                        {
                            string circleId = circle.ObjectId.ToString() + "c";
                            double circleRadius = circle.Radius;

                            // Check if circle node already exists
                            Node circleNode = FindOrCreateNode(circle.Center, "geometry", circleId, "circle", nodesList, circleRadius);

                        }
                    }

                    // debugging
                    ed.WriteMessage("\n2-Overlaps counting...");
                    // Find overlap edges
                    FindOverlaps(nodesList, edgesList);


                    // debugging
                    ed.WriteMessage("\n3-Node level calculations...");
                    // Calculate node level statistics
                    NodeLevelCalculations.CalculateNodeDegree(nodesList, edgesList);
                    NodeLevelCalculations.CalculateNodeEigenvectorCentrality(nodesList, edgesList);
                    NodeLevelCalculations.CalculateBetweennessCentrality(nodesList, edgesList);
                    NodeLevelCalculations.CalculateClosenessCentrality(nodesList, edgesList);
                    NodeLevelCalculations.CalculateClusteringCoefficient(nodesList, edgesList);
                    NodeLevelCalculations.CalculatePageRank(nodesList, edgesList);

                    // debugging
                    ed.WriteMessage("\n4-Graph creation...");
                    // Define the graph object
                    Graph graph = new Graph { Edges = edgesList, Nodes = nodesList };

                    // Calculate graph level statistics
                    GraphLevelCalculations.CalculateAverageDegree(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateGraphDensity(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateGraphDiameter(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateGraphKernelDegree(nodesList, graph);
                    GraphLevelCalculations.CalculateKernelDegreeDistribution(nodesList, graph);
                    GraphLevelCalculations.CalculateConnectedComponents(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateAveragePathLength(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateAssortivityCoefficient(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateTransitivity(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateSpectralRadius(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateAlgebraicConnectivity(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateEigenvalueDistribution(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateModularity(nodesList, edgesList, graph);
                    GraphLevelCalculations.CalculateEntropyOfDegreeDistribution(nodesList, graph);
                    GraphLevelCalculations.CalculateNodeConnectivityAlternative(nodesList, edgesList, graph);

                    // debugging
                    ed.WriteMessage("\n5-Serializing graph object...");
                    // Serialize the graph object to JSON
                    string json = JsonConvert.SerializeObject(graph, Formatting.Indented);

                    // Acquire the file path of active document
                    string docPath = doc.Name;
                    string docDir = Path.GetDirectoryName(docPath);

                    // Define the output file path
                    string outputPath = Path.Combine(docDir, $"{doc.Name}_graph.json");

                    // debugging
                    ed.WriteMessage("\n6-Writing JSON to file...");
                    // Write the JSON to the output file
                    File.WriteAllText(outputPath, json);

                    // Commit the transaction
                    tr.Commit();
                }
                catch (System.Exception ex)
                {
                    ed.WriteMessage($"\nError: {ex.Message}");
                }
            }
        }

        // Helper method - Find an existing node or create a new one
        private Node FindOrCreateNode(Point3d pt, string type, string id, string label, List<Node> nodesList, double size = 0)
        {
            // Find an existing node at the same pt and having the same label
            Node existingNode = nodesList.FirstOrDefault(n => n.X == pt.X && n.Y == pt.Y && n.Z == pt.Z && n.label == label);

            if (existingNode != null)
            {
                return existingNode;
            }

            // If no existing node found, create a new node
            Node newNode;

            if (size > 0)
            {
                newNode = new Node(pt, type, id, label, size);
            }
            else
            {
                newNode = new Node(pt, type, id, label);
            }
            nodesList.Add(newNode);
            return newNode;
        }

        // Helper method - Find overlapping nodes in nodesList and establish a "overlap" edge between each of them
        private void FindOverlaps(List<Node> nodesList, List<Edge> edgesList)
        {
            // Iterate through all nodes in the list
            for (int i = 0; i < nodesList.Count; i++)
            {
                // Compare each node with all other nodes
                for (int j = i + 1; j < nodesList.Count; j++)
                {
                    if (nodesList[i].X == nodesList[j].X && nodesList[i].Y == nodesList[j].Y && nodesList[i].Z == nodesList[j].Z)
                    {
                        string overlapId = nodesList[i].id + nodesList[j].id;
                        Edge overlapEdge = new Edge(nodesList[i], nodesList[j], "geometry", overlapId, "overlap");
                        edgesList.Add(overlapEdge);
                    }
                }
            }
        }

    }

}
