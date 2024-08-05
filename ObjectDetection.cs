using Microsoft.ML;
using Microsoft.ML.Data;

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Autodesk.AutoCAD.Runtime;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.DatabaseServices;
using Autodesk.AutoCAD.EditorInput;
using Autodesk.AutoCAD.Geometry;


namespace Graphit
{
    public class CadDataExtractor : BaseCommand
    {

        public static string csvPath;

        // Command that extracts training data from the drawing

        [CommandMethod("ExtractTrainingData")]
        public void ExtractTrainingData()
        {
            using (Transaction tr = db.TransactionManager.StartTransaction())
            {
                BlockTable bt = tr.GetObject(db.BlockTableId, OpenMode.ForRead) as BlockTable;
                BlockTableRecord btr = tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead) as BlockTableRecord;

                // Create a list to store the lines data
                List<string> linesList = new List<string>();
                linesList.Add("Type,X,Y,Layer,Label");

                foreach (ObjectId objId in btr)
                {
                    Entity ent = tr.GetObject(objId, OpenMode.ForRead) as Entity;
                    string type  = ent.GetType().Name;
                    string layer = ent.Layer;
                    string label = GetLabel(ent); 

                    if (ent is Line line)
                    { 
                        string lineData = $"{type},{line.StartPoint.X},{line.StartPoint.Y},{layer},{label}"; 
                        linesList.Add(lineData);
                    }

                    else if (ent is Circle circle)
                    {
                        string circleData = $"{type},{circle.Center.X},{circle.Center.Y},{layer},{label}";
                        linesList.Add(circleData);
                    }

                    // TODO : Add similar code blocks for other entity types

                }


                // Save the data to a CSV file to the path where the drawing is located
                string drawingPath = db.Filename;
                string drawingFolder = Path.GetDirectoryName(drawingPath);
                csvPath = Path.Combine(drawingFolder, "training_data.csv");
                File.WriteAllLines(csvPath, linesList);
                /*
                // Save the data to a CSV file to the path where the executable is located
                File.WriteAllLines("training_data.csv", linesList);
                */
                tr.Commit();
            }
        }

        private string GetLabel(Entity entity)
        {
            // Final labelling supposed to be made manually. 
            // If not so, implement logic here to automatically label the entities.
            return "correct_label";
        }

        /////////////////////////////////////////////////////
        ///
        public class OutlierDetection : BaseCommand
        {
            private static PredictionEngine<DataPoint, Prediction> predictionEngine;

            [CommandMethod("DetectOutliers")]
            public void DetectOutliers()
            {
                LoadModel();

                using (Transaction tr = db.TransactionManager.StartTransaction())
                {
                    BlockTable bt = tr.GetObject(db.BlockTableId, OpenMode.ForRead) as BlockTable;
                    BlockTableRecord btr = tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead) as BlockTableRecord;

                    foreach (ObjectId objId in btr)
                    {
                        Entity ent = tr.GetObject(objId, OpenMode.ForRead) as Entity;
                        string type = ent.GetType().Name;
                        string layer = ent.Layer;

                        if (ent is Line line)
                        {
                            var dataPoint = new DataPoint
                            {
                                Type = type,
                                X = (float)line.StartPoint.X,
                                Y = (float)line.StartPoint.Y,
                                Layer = layer
                            };

                            var prediction = predictionEngine.Predict(dataPoint);

                            if (prediction.PredictedLabel != "correct_label")
                            {
                                ed.WriteMessage($"Outlier detected: {type} on layer {layer}");
                            }
                        }

                        else if (ent is Circle circle)
                        {
                            var dataPoint = new DataPoint
                            {
                                Type = "Circle",
                                X = (float)circle.Center.X,
                                Y = (float)circle.Center.Y,
                                Layer = circle.Layer
                            };

                            var prediction = predictionEngine.Predict(dataPoint);

                            if (prediction.PredictedLabel != "correct_label")
                            {
                                ed.WriteMessage($"Outlier detected: {type} on layer {layer}");
                            }
                        }
                        
                        // TODO : Add similar code blocks for other entity types
                    }
                }


            }
            private void LoadModel()
            {
                if (predictionEngine == null)
                {
                    var mlContext = new MLContext();

                    // Load the model from the path where the drawing file is located
                    var modelPath = csvPath;
                    var model = mlContext.Model.Load(modelPath, out var modelInputSchema);

                    // For Debugging - Logging the schema data
                    Console.WriteLine("Model Input Schema: ");
                    foreach (var column in modelInputSchema)
                    {
                        Console.WriteLine($"Name: {column.Name}, Type: {column.Type.RawType}");
                    }

                    predictionEngine = mlContext.Model.CreatePredictionEngine<DataPoint, Prediction>(model);
                }
            }

            public class Prediction
            {
                public string PredictedLabel;
                public float[] Score;
            }
        }
    }
}


