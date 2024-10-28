# 1. Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("LogisticRegressionExample") \
    .getOrCreate()

data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

data_indexed = status_type_indexer.fit(data).transform(data)
data_indexed = status_published_indexer.fit(data_indexed).transform(data_indexed)

assembler = VectorAssembler(inputCols=["status_type_ind", "status_published_ind"], outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="status_type_ind")
lr.setMaxIter(100).setRegParam(0.1).setElasticNetParam(0.5)

pipeline = Pipeline(stages=[assembler, lr])

train_data, test_data = data_indexed.randomSplit([0.8, 0.2])

pipeline_model = pipeline.fit(train_data)

predictions = pipeline_model.transform(test_data)

predictions.select("status_type_ind", "prediction", "probability").show(5)

evaluator = MulticlassClassificationEvaluator(labelCol="status_type_ind", predictionCol="prediction")

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
print(f"Accuracy: {accuracy}")

precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
print(f"Precision: {precision}")

recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
print(f"Recall: {recall}")

f1 = evaluator.setMetricName("f1").evaluate(predictions)
print(f"F1 measure: {f1}")

spark.stop()
