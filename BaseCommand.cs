using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.DatabaseServices;
using Autodesk.AutoCAD.EditorInput;
using Autodesk.AutoCAD.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Graphit
{
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

        // Derived metrics
        public int degree { get; set; }
        public double centralityEigenvector { get; set; }
        public double centralityBetweenness { get; set; }
        public double centralityCloseness { get; set; }
        public double clusteringCoefficient { get; set; }


        public Node(Point3d pt, string type, string id, string label)
        {
            X = pt.X;
            Y = pt.Y;
            Z = pt.Z;
            this.type = type; // --- geometry
            this.id = id;
            this.label = label; // --- line, circle
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
        public string label { get; set; } // --- connection, overlap
        public string type { get; set; }

        public Edge(Node start, Node end, string type, string id, string label)
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

        // Derived metrics
        public double averageDegree { get; set; }
        public double density { get; set; }
        public Dictionary<int, int> kernelDegree { get; set; }
    }

}
