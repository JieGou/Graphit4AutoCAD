using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace Graphit
{
    public class ModelTraining
    {
        public static void TrainModel()
        {
            // Create a new ML context
            var mlContext = new MLContext(seed: 0);

            // Load the data
            var dataPath = @"C:\path\to\dataset.csv";
            var data = mlContext.Data.LoadFromTextFile<DataPoint>(dataPath, separatorChar: ',', hasHeader: true);

            // Split data into training and testing datasets
            var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Define data preparation and training pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Type")) 
                .Append(mlContext.Transforms.Concatenate("Features", "Type", "X", "Y", "Layer")) 
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label")) 
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

            // Save the model
            mlContext.Model.Save(model, splitData.TrainSet.Schema, @"C:\path\to\model.zip");
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
