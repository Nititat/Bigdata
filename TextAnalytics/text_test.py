# นำเข้าชุดไลบรารีที่จำเป็น
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# สร้าง SparkSession
spark = SparkSession.builder.appName("TextAnalytics").getOrCreate()

# โหลดข้อมูลจากไฟล์ (คุณอาจต้องแทนที่ 'reviews_rated.csv' ด้วยชื่อไฟล์ที่ถูกต้อง)
data = spark.read.csv("reviews_rated.csv", header=True, inferSchema=True)

# เลือกเฉพาะคอลัมน์ Review Text และ Rating และแปลง Rating เป็น Integer
data = data.select(data["Review Text"].alias("review_text"), data["Rating"].cast(IntegerType()).alias("rating"))
data = data.na.drop()  # ลบข้อมูลที่มีค่าว่างออกไป
data.show(5)

# แยกคำออกจากข้อความ (Tokenizer)
tokenizer = Tokenizer(inputCol="review_text", outputCol="words")

# ลบคำที่ไม่สำคัญ (StopWordsRemover)
stopword_remover = StopWordsRemover(inputCol="words", outputCol="meaningful_words")

# แปลงคำเป็นคุณลักษณะโดยใช้ Term Frequency (HashingTF)
hashing_tf = HashingTF(inputCol="meaningful_words", outputCol="features")

# สร้าง Pipeline ที่รวมขั้นตอนต่าง ๆ
pipeline = Pipeline(stages=[tokenizer, stopword_remover, hashing_tf])

# แบ่งข้อมูลเป็นชุดฝึกอบรมและชุดทดสอบ
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# ฟิต Pipeline กับข้อมูลฝึกอบรม
pipeline_model = pipeline.fit(train_data)

# แปลงข้อมูลฝึกอบรมและทดสอบด้วย Pipeline
train_transformed = pipeline_model.transform(train_data)
test_transformed = pipeline_model.transform(test_data)

# แสดงตัวอย่างข้อมูลที่ถูกแปลง
train_transformed.select("meaningful_words", "features", "rating").show(5)

# สร้าง Logistic Regression Model
log_reg = LogisticRegression(labelCol="rating", featuresCol="features")

# ฟิต Logistic Regression กับข้อมูลฝึกอบรม
log_reg_model = log_reg.fit(train_transformed)

# ทำนายผลบนข้อมูลทดสอบ
predictions = log_reg_model.transform(test_transformed)

# แสดงผลลัพธ์ที่ถูกทำนาย
predictions.select("meaningful_words", "rating", "prediction").show(5)

# ประเมินผลความแม่นยำของโมเดล
evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
