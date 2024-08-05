using Autodesk.AutoCAD.Runtime;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.IO;

namespace Graphit
{
    public class ModelTraining : BaseCommand
    {
        // Command that trains the model
        // TODO : properly handle the case where there are not enough classes in the training data.

        [CommandMethod("TrainModel")]
        public void TrainModel()
        {
            try
            {
                // Create a new ML context
                MLContext mlContext = new MLContext(seed: 0);

                // Load the data from csvPath variable of the CadDataExtractor class of ObjectDetection.cs file
                string dataPath = CadDataExtractor.csvPath;
                var data = mlContext.Data.LoadFromTextFile<DataPoint>(dataPath, separatorChar: ',', hasHeader: true); // --- Change the separatorChar if needed

                // Split data into training and testing datasets
                var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

                // Define data preparation and training pipeline
                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("TypeEncoded", "Type"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("LayerEncoded", "Layer"))
                    .Append(mlContext.Transforms.Concatenate("NumericFeatures", "X", "Y"))
                    .Append(mlContext.Transforms.Concatenate("Features", "TypeEncoded", "LayerEncoded", "NumericFeatures"))
                    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                // Train the model
                var model = pipeline.Fit(splitData.TrainSet);

                // Evaluate the model
                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

                // Print evaluation metrics
                Console.WriteLine($"Log-loss: {metrics.LogLoss}");
                Console.WriteLine($"Macro accuracy: {metrics.MacroAccuracy}");
                Console.WriteLine($"Micro accuracy: {metrics.MicroAccuracy}");

                // Save the model to path of drawing file
                mlContext.Model.Save(model, splitData.TrainSet.Schema, Path.ChangeExtension(dataPath, ".zip"));
            }
            catch (System.Exception ex)
            {
                ed.WriteMessage($"\nSystem Error: {ex.Message}");
            }
        }
    }

    // Define the data schema
    public class DataPoint
    {
        [LoadColumn(0)] public string Type;
        [LoadColumn(1)] public float X;
        [LoadColumn(2)] public float Y;
        [LoadColumn(3)] public string Layer;
        [LoadColumn(4)] public string Label;
    }
}
