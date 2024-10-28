from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("FBLiveTH").getOrCreate()

data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

status_type_encoder = OneHotEncoder(inputCol="status_type_ind", outputCol="status_type_encoded")
status_published_encoder = OneHotEncoder(inputCol="status_published_ind", outputCol="status_published_encoded")

assembler = VectorAssembler(inputCols=["status_type_encoded", "status_published_encoded"], outputCol="features")

pipeline = Pipeline(stages=[status_type_indexer, status_published_indexer, status_type_encoder, status_published_encoder, assembler])

pipeline_model = pipeline.fit(data)

transformed_data = pipeline_model.transform(data)

train_data, test_data = transformed_data.randomSplit([0.8, 0.2])

decision_tree = DecisionTreeClassifier(labelCol="status_type_ind", featuresCol="features")

decision_tree_model = decision_tree.fit(train_data)

predictions = decision_tree_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="status_type_ind", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

test_error = 1.0 - accuracy

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Measure: {f1}")
print(f"Test Error: {test_error}")