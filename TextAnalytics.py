#โค้ดนี้เป็นการวิเคราะห์และจำแนกข้อความโดยใช้โมเดลการถดถอยโลจิสติก (Logistic Regression) ในการทำนาย คะแนน (Rating) 
# นำเข้าไลบรารีที่จำเป็นสำหรับการทำงานกับ Spark, การจัดการข้อมูล, และการสร้างโมเดลการทำนาย
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# สร้าง SparkSession สำหรับเริ่มต้นการประมวลผลข้อมูลใน Spark
spark = SparkSession.builder.appName("TextAnalytics").getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV (แทนที่ 'reviews_rated.csv' ด้วยไฟล์ข้อมูลจริงของคุณ)
data = spark.read.csv("reviews_rated.csv", header=True, inferSchema=True)

# เลือกคอลัมน์ "Review Text" และ "Rating" จากข้อมูล และแปลงประเภทของ "Rating" เป็น IntegerType
# โดยทำการตั้งชื่อใหม่ให้คอลัมน์ "Review Text" เป็น "review_text"
data = data.select(data["Review Text"].alias("review_text"), data["Rating"].cast(IntegerType()).alias("rating"))

# ลบแถวที่มีค่าว่าง (missing values) ออก
data = data.na.drop()

# แสดงข้อมูล 5 แถวแรกเพื่อดูตัวอย่างข้อมูล
data.show(5)

# Tokenizer: แยกข้อความรีวิว (review_text) ออกเป็นคำ (words)
tokenizer = Tokenizer(inputCol="review_text", outputCol="words")

# StopWordsRemover: ลบคำทั่วไป (stop words) ที่ไม่จำเป็นในการวิเคราะห์ เช่น "the", "is", "and"
stopword_remover = StopWordsRemover(inputCol="words", outputCol="meaningful_words")

# HashingTF: แปลงคำที่เหลืออยู่ให้เป็นตัวเลขคุณลักษณะ (features) โดยใช้เทคนิค Term Frequency (TF)
hashing_tf = HashingTF(inputCol="meaningful_words", outputCol="features")

# สร้าง Pipeline เพื่อจัดการกระบวนการแปลงข้อมูลตามลำดับ: Tokenizer -> StopWordsRemover -> HashingTF
pipeline = Pipeline(stages=[tokenizer, stopword_remover, hashing_tf])

# แบ่งข้อมูลออกเป็นชุดฝึก (training set) และชุดทดสอบ (testing set) ในสัดส่วน 80:20
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# ใช้ Pipeline กับชุดฝึก (training set) เพื่อแปลงข้อมูลตามลำดับที่กำหนดใน Pipeline
pipeline_model = pipeline.fit(train_data)

# แปลงข้อมูลในชุดฝึก (training set) และชุดทดสอบ (testing set) โดยใช้ Pipeline ที่ฝึกแล้ว
train_transformed = pipeline_model.transform(train_data)
test_transformed = pipeline_model.transform(test_data)

# แสดงคำที่มีความหมาย (meaningful_words) คุณลักษณะ (features) และเรตติ้ง (rating) ของข้อมูลฝึกที่แปลงแล้ว
train_transformed.select("meaningful_words", "features", "rating").show(5)

# สร้าง Logistic Regression โมเดล โดยตั้งค่าคอลัมน์ label เป็น "rating" และคอลัมน์ features เป็น "features"
log_reg = LogisticRegression(labelCol="rating", featuresCol="features")

# ฝึก Logistic Regression โมเดลด้วยข้อมูลฝึก (training set)
log_reg_model = log_reg.fit(train_transformed)

# ใช้โมเดลที่ฝึกแล้วในการทำนายข้อมูลในชุดทดสอบ (testing set)
predictions = log_reg_model.transform(test_transformed)

# แสดงคำที่มีความหมาย (meaningful_words), เรตติ้ง (rating), และผลลัพธ์การทำนาย (prediction) ของข้อมูลทดสอบ
predictions.select("meaningful_words", "rating", "prediction").show(5)

# ประเมินความถูกต้อง (accuracy) ของโมเดลด้วยการใช้ MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")

# คำนวณและแสดงค่า Accuracy ของโมเดล
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
