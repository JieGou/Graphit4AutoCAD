// AutoCad to graph conversion tool

using Autodesk.AutoCAD.Runtime; 
using Autodesk.AutoCAD.ApplicationServices; 
using Autodesk.AutoCAD.DatabaseServices; 
using Autodesk.AutoCAD.EditorInput; 
using Autodesk.AutoCAD.Geometry;

using Newtonsoft.Json;

using System.Collections.Generic;
using System.IO;

namespace Graphit
{
    // Base class for common declarations
    public abstract class BaseCommand
    {
        protected Document doc;
        protected Database db;
        protected Editor ed;

        protected BaseCommand()
        {
            doc = Application.DocumentManager.MdiActiveDocument;
            db = doc.Database;
            ed = doc.Editor;
        }
    }

    public class Graphit : BaseCommand
    {
        // Extract Autocad objects to graph database

        [CommandMethod("ExtractToGraph")]
        public void ExtractToGraph()
        {

            // Start a transaction
            using (Transaction tr = db.TransactionManager.StartTransaction())
            {
                // Open the Block table & Block table record for read
                BlockTable bt = tr.GetObject(db.BlockTableId, OpenMode.ForRead) as BlockTable;
                BlockTableRecord btr = tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead) as BlockTableRecord;

                List<Node> nodes = new List<Node>();
                List<Edge> edges = new List<Edge>();

                // Iterate through all the objects in Model space
                foreach (ObjectId objId in btr)
                {
                    Entity ent = tr.GetObject(objId, OpenMode.ForRead) as Entity;

                    if (ent is Line line)
                    {
                        string lineId = line.ObjectId.ToString();
                        Node startNode = new Node(line.StartPoint, "position", lineId, "line");
                        Node endNode = new Node(line.EndPoint, "position", lineId, "line");
                        Edge edge = new Edge(startNode, endNode, "Line", lineId, "connection");
                        nodes.Add(startNode);
                        nodes.Add(endNode);
                        edges.Add(edge);
                    }

                    if (ent is Circle circle)
                    {
                        string circleId = circle.ObjectId.ToString();
                        double circleRadius = circle.Radius;
                        Node circleNode = new Node(circle.Center, "position", circleId, "circle", circleRadius);
                        nodes.Add(circleNode);
                    }

                    // Define the graph object
                    Graph graph = new Graph { Edges = edges, Nodes = nodes };

                    // Serialize the graph object to JSON
                    string json = JsonConvert.SerializeObject(graph, Formatting.Indented);

                    // Acquire the file path of active document
                    string docPath = doc.Name;
                    string docDir = Path.GetDirectoryName(docPath);

                    // Define the output file path
                    string outputPath = Path.Combine(docDir, "graph.json");

                    // Write the JSON to the output file
                    File.WriteAllText(outputPath, json);
                }
                
            }
        }
    }

    public class Node
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
        public string type { get; set; }
        public string id { get; set; }
        public string label { get; set; }
        public string color { get; set; }
        public string shape { get; set; }
        public double size { get; set; }

        public Node(Point3d pt, string type, string id, string label)
        {
            X = pt.X;
            Y = pt.Y;
            Z = pt.Z;
            this.type = type;
            this.id = id;
            this.label = label;
        }

        // Node overload 01 - Circle
        public Node(Point3d pt, string type, string id, string label, double size)
        {
            X = pt.X;
            Y = pt.Y;
            Z = pt.Z;
            this.type = type;
            this.id = id;
            this.label = label;
            this.size = size;
        }
    }

    public class Edge
    {
        public Node start { get; set; }
        public Node end { get; set; }
        public string id { get; set; }
        public string label { get; set; }
        public string type { get; set; }

        public Edge (Node start, Node end, string type, string id, string label)
        {
            this.start = start;
            this.end = end;
            this.type = type;
            this.id = id;
            this.label = label;
        }
    }

    public class Graph
    {
        public List<Node> Nodes { get; set; }
        public List<Edge> Edges { get; set; }
    }
}
