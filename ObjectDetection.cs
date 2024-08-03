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
        [CommandMethod("ExtractTrainingData")]
        public void ExtractTrainingData()
        {
            using (Transaction tr = db.TransactionManager.StartTransaction())
            {
                BlockTable bt = tr.GetObject(db.BlockTableId, OpenMode.ForRead) as BlockTable;
                BlockTableRecord btr = tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead) as BlockTableRecord;

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

                    // TODO : Add similar code clocks fot other entity types

                }
                
                File.WriteAllLines("training_data.csv", linesList);
                tr.Commit();

            }

        }
        private string GetLabel(Entity entity)
        {
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
                    var modelPath = @"C:\path\to\model.zip";
                    var model = mlContext.Model.Load(modelPath, out var modelInputSchema);
                    predictionEngine = mlContext.Model.CreatePredictionEngine<DataPoint, Prediction>(model);
                }
            }

            public class DataPoint
            {
                public string Type;
                public float X;
                public float Y;
                public string Layer;
            }

            public class Prediction
            {
                public string PredictedLabel;
                public float[] Score;
            }
        }
    }
}


