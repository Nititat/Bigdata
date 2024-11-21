from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("BookRecommendationALS") \
    .getOrCreate()

data = spark.read.csv('book_ratings.csv', header=True, inferSchema=True)

data.printSchema()
data.show(5)

# Define ALS model
als = ALS(
    maxIter=10,  # Number of iterations
    userCol="user_id",  # Column for user ID
    itemCol="book_id",  # Column for item (book) ID
    ratingCol="rating",  # Column for ratings
    coldStartStrategy="drop"  # Drop rows with NaN predictions
)

# Fit the ALS model
model = als.fit(data)

# Predict ratings using the trained model
predictions = model.transform(data)

# Define RegressionEvaluator
evaluator = RegressionEvaluator(
    metricName="rmse",  # Root Mean Squared Error
    labelCol="rating",  # Column for the actual ratings
    predictionCol="prediction"  # Column for predicted ratings
)

# Evaluate the model
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Show predictions for User ID = 53
user_id = 53
user_predictions = predictions.filter(col("user_id") == user_id)
user_predictions = user_predictions.select("book_id", "user_id", "rating", "prediction").orderBy(col("prediction").desc())
user_predictions.show(truncate=False)

# Show 5 recommended books for all users
user_recommendations = model.recommendForAllUsers(5)
user_recommendations.show(truncate=False)

# Show 5 recommended users for all books
item_recommendations = model.recommendForAllItems(5)
item_recommendations.show(truncate=False)

# Stop SparkSession
spark.stop()