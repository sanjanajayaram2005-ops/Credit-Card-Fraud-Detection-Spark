from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Credit Card Fraud Detection") \
    .master("local[*]") \
    .getOrCreate()

# Load the credit card dataset
df = spark.read.csv(
    "creditcard.csv",
    header=True,
    inferSchema=True
)

print("Dataset loaded successfully!")

print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

print("\nDataset columns:")
print(df.columns)

print("\nDataset schema:")
df.printSchema()

print("\nFirst 5 rows:")
df.show(5)

print("\nTransaction class distribution:")
df.groupBy("Class").count().show()

print("\nMissing values:")
from pyspark.sql.functions import col, sum

df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show()

print("\nDuplicate rows:")
duplicate_count = df.count() - df.dropDuplicates().count()
print("Number of duplicate rows:", duplicate_count)

# Remove duplicate rows
df = df.dropDuplicates()

print("Rows after removing duplicates:", df.count())
print("\nFraud distribution after cleaning:")
df.groupBy("Class").count().show()

from pyspark.ml.feature import VectorAssembler

# Features used for fraud detection
feature_columns = [col for col in df.columns if col != "Class"]

# Combine all input columns into a single feature vector
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

df = assembler.transform(df)

print("\nFeature vector created successfully!")
df.select("features", "Class").show(5, truncate=False)

# Split data into training and testing sets
train_data, test_data = df.select("features", "Class").randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining data count:", train_data.count())
print("Testing data count:", test_data.count())

from pyspark.ml.classification import LogisticRegression

# Create Logistic Regression model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="Class"
)

# Train the model
lr_model = lr.fit(train_data)

print("\nLogistic Regression model trained successfully!")

# Make predictions on test data
predictions = lr_model.transform(test_data)

print("\nPredictions generated successfully!")
predictions.select("Class", "prediction", "probability").show(10, truncate=False)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Model evaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="Class",
    predictionCol="prediction"
)

accuracy = evaluator.evaluate(
    predictions, {evaluator.metricName: "accuracy"}
)

precision = evaluator.evaluate(
    predictions,
    {evaluator.metricName: "precisionByLabel", evaluator.metricLabel: 1.0}
)

recall = evaluator.evaluate(
    predictions,
    {evaluator.metricName: "recallByLabel", evaluator.metricLabel: 1.0}
)
f1 = evaluator.evaluate(
    predictions,
    {evaluator.metricName: "fMeasureByLabel", evaluator.metricLabel: 1.0}
)
# ROC-AUC
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="Class",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

roc_auc = binary_evaluator.evaluate(predictions)

# PR-AUC
pr_evaluator = BinaryClassificationEvaluator(
    labelCol="Class",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

pr_auc = pr_evaluator.evaluate(predictions)

print("\n===== MODEL EVALUATION =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)
print("PR-AUC   :", pr_auc)

# Confusion matrix
print("\n===== CONFUSION MATRIX =====")
predictions.groupBy("Class", "prediction").count().orderBy("Class", "prediction").show()

# ================= RANDOM FOREST MODEL =================

from pyspark.ml.classification import RandomForestClassifier

# Create Random Forest model
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="Class",
    numTrees=100,
    seed=42
)

# Train the model
rf_model = rf.fit(train_data)

print("\nRandom Forest model trained successfully!")

# Make predictions on test data
rf_predictions = rf_model.transform(test_data)

print("Random Forest predictions generated successfully!")

# Show sample predictions
rf_predictions.select(
    "Class",
    "prediction",
    "probability"
).show(10, truncate=False)

# ================= RANDOM FOREST EVALUATION =================

rf_evaluator = MulticlassClassificationEvaluator(
    labelCol="Class",
    predictionCol="prediction"
)

rf_accuracy = rf_evaluator.evaluate(
    rf_predictions,
    {rf_evaluator.metricName: "accuracy"}
)

rf_precision = rf_evaluator.evaluate(
    rf_predictions,
    {rf_evaluator.metricName: "weightedPrecision"}
)

rf_recall = rf_evaluator.evaluate(
    rf_predictions,
    {rf_evaluator.metricName: "weightedRecall"}
)

rf_f1 = rf_evaluator.evaluate(
    rf_predictions,
    {rf_evaluator.metricName: "f1"}
)

# ROC-AUC
rf_binary_evaluator = BinaryClassificationEvaluator(
    labelCol="Class",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

rf_roc_auc = rf_binary_evaluator.evaluate(rf_predictions)

# PR-AUC
rf_pr_evaluator = BinaryClassificationEvaluator(
    labelCol="Class",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

rf_pr_auc = rf_pr_evaluator.evaluate(rf_predictions)

print("\n===== RANDOM FOREST EVALUATION =====")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)
print("ROC-AUC  :", rf_roc_auc)
print("PR-AUC   :", rf_pr_auc)

# ================= MODEL COMPARISON =================

print("\n===== MODEL COMPARISON =====")

comparison_data = [
    ("Logistic Regression", accuracy, precision, recall, f1, roc_auc, pr_auc),
    ("Random Forest", rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc, rf_pr_auc)
]

comparison_df = spark.createDataFrame(
    comparison_data,
    ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
)

comparison_df.show(truncate=False)

# Save model comparison results
# Save results using Pandas instead of Spark CSV writer
import pandas as pd
from pyspark.ml.functions import vector_to_array

# Save model comparison
comparison_pd = comparison_df.toPandas()
comparison_pd.to_csv("model_comparison.csv", index=False)

# Save Random Forest predictions
rf_output = rf_predictions.select(
    "Class",
    "prediction",
    vector_to_array("probability")[1].alias("fraud_probability")
)

rf_output_pd = rf_output.toPandas()
rf_output_pd.to_csv("fraud_predictions.csv", index=False)

print("Results saved successfully!")

spark.stop()

import os
os._exit(0)