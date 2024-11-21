# นำเข้าไลบรารีที่จำเป็นจาก PySpark สำหรับการสร้าง SparkSession, การจัดการข้อมูล, การแปลงข้อมูล, และการสร้างโมเดล
from pyspark.sql import SparkSession  # ใช้สำหรับการสร้าง SparkSession
from pyspark.sql.types import IntegerType  # ใช้สำหรับการแปลงชนิดข้อมูลให้เป็น Integer
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF  # ใช้สำหรับการจัดการข้อความ (text processing)
from pyspark.ml.classification import LogisticRegression  # ใช้สำหรับการจำแนกประเภทด้วย Logistic Regression
from pyspark.ml import Pipeline  # ใช้สำหรับการรวมขั้นตอนต่าง ๆ ของการทำงานเป็นขั้นตอนเดียว (pipeline)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  # ใช้สำหรับประเมินประสิทธิภาพของโมเดลที่มีหลายคลาส (multiclass)

# สร้าง SparkSession เพื่อเริ่มการทำงานกับ PySpark
spark = SparkSession.builder.appName("TextAnalytics").getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV (แทนที่ 'reviews_rated.csv' ด้วยชื่อไฟล์ที่คุณใช้งานจริง)
# data เป็นตัวแปรที่เก็บข้อมูลรีวิวที่อ่านมาจากไฟล์ CSV
# header=True หมายความว่าบรรทัดแรกของไฟล์ CSV เป็นหัวข้อคอลัมน์
# inferSchema=True เพื่อให้ Spark เดาชุดข้อมูลว่ามีชนิดข้อมูลเป็นอะไร
data = spark.read.csv("reviews_rated.csv", header=True, inferSchema=True)

# เลือกเฉพาะคอลัมน์ "Review Text" และ "Rating" จาก DataFrame
# เปลี่ยนชื่อคอลัมน์ "Review Text" เป็น "review_text" และแปลงคอลัมน์ "Rating" เป็นชนิด Integer
data = data.select(data["Review Text"].alias("review_text"), data["Rating"].cast(IntegerType()).alias("rating"))

# ลบข้อมูลที่มีค่าว่างออก (na.drop() ลบแถวที่มีข้อมูลว่าง)
data = data.na.drop()

# แสดงข้อมูล 5 แถวแรกของ DataFrame เพื่อดูว่าข้อมูลถูกต้องหรือไม่
data.show(5)

# Tokenizer: แบ่งข้อความเป็นคำ (แยกแต่ละคำในข้อความออกมา)
# tokenizer เป็นตัวแปรที่เก็บขั้นตอนการแปลงข้อความในคอลัมน์ "review_text" ให้กลายเป็นคำในคอลัมน์ "words"
tokenizer = Tokenizer(inputCol="review_text", outputCol="words")

# StopWordsRemover: ลบคำที่ไม่สำคัญ เช่น คำว่า "the", "is", "and" เป็นต้น
# stopword_remover เป็นตัวแปรที่เก็บขั้นตอนการลบคำทั่วไป (stop words) ออกจากคอลัมน์ "words"
# ผลลัพธ์จะเก็บในคอลัมน์ใหม่ชื่อ "meaningful_words"
stopword_remover = StopWordsRemover(inputCol="words", outputCol="meaningful_words")

# HashingTF: แปลงคำให้กลายเป็นฟีเจอร์โดยใช้ Term Frequency (TF)
# hashing_tf เป็นตัวแปรที่เก็บขั้นตอนการแปลงคำในคอลัมน์ "meaningful_words" ให้กลายเป็นเวกเตอร์ฟีเจอร์ในคอลัมน์ "features"
hashing_tf = HashingTF(inputCol="meaningful_words", outputCol="features")

# สร้าง Pipeline เพื่อนำขั้นตอนการแปลงข้อมูล (tokenizer, stopword_remover, hashing_tf) มารวมกัน
# pipeline เป็นตัวแปรที่เก็บขั้นตอนการแปลงข้อมูลทั้งหมดเป็นลำดับ (ขั้นตอนการแปลงจะถูกใช้ตามลำดับที่ระบุใน stages)
pipeline = Pipeline(stages=[tokenizer, stopword_remover, hashing_tf])

# แบ่งข้อมูลออกเป็นชุดฝึก (train_data) และชุดทดสอบ (test_data)
# train_data และ test_data เป็นข้อมูลที่ถูกแบ่งเป็น 80% สำหรับการฝึกโมเดล และ 20% สำหรับการทดสอบ
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)  # seed=1234 ใช้เพื่อให้การแบ่งข้อมูลได้ผลลัพธ์ที่ทำซ้ำได้

# ฝึก Pipeline ด้วยข้อมูลชุดฝึก (train_data)
# pipeline_model เป็นตัวแปรที่เก็บโมเดลที่ถูกฝึกเสร็จแล้ว
pipeline_model = pipeline.fit(train_data)

# แปลงข้อมูลชุดฝึกและชุดทดสอบด้วย pipeline ที่ถูกฝึกแล้ว
# train_transformed และ test_transformed เป็นข้อมูลที่ถูกแปลงแล้วตามขั้นตอนใน pipeline
train_transformed = pipeline_model.transform(train_data)
test_transformed = pipeline_model.transform(test_data)

# แสดงคอลัมน์ "meaningful_words", "features", และ "rating" ของข้อมูลที่ถูกแปลงแล้ว
train_transformed.select("meaningful_words", "features", "rating").show(5)

# สร้างโมเดล Logistic Regression สำหรับการจัดประเภท (classification)
# log_reg เป็นตัวแปรที่เก็บโมเดล Logistic Regression ซึ่งจะใช้ข้อมูลในคอลัมน์ "features" เพื่อทำนายคอลัมน์ "rating"
log_reg = LogisticRegression(labelCol="rating", featuresCol="features")

# ฝึกโมเดล Logistic Regression ด้วยข้อมูลชุดฝึกที่ถูกแปลงแล้ว
# log_reg_model เป็นตัวแปรที่เก็บโมเดล Logistic Regression ที่ถูกฝึกเสร็จแล้ว
log_reg_model = log_reg.fit(train_transformed)

# ใช้โมเดลที่ฝึกแล้วทำนายข้อมูลในชุดทดสอบ
# predictions เป็นตัวแปรที่เก็บผลลัพธ์ที่ถูกทำนายจากข้อมูลชุดทดสอบ
predictions = log_reg_model.transform(test_transformed)

# แสดงผลลัพธ์ของการทำนาย โดยเลือกคอลัมน์ "meaningful_words" (คำที่มีความหมาย), "rating" (ค่าจริง), และ "prediction" (ค่าทำนาย)
predictions.select("meaningful_words", "rating", "prediction").show(5)

# สร้างตัวประเมินผลลัพธ์ของโมเดล Logistic Regression
# evaluator เป็นตัวแปรที่เก็บตัวประเมินผลลัพธ์สำหรับการจำแนกประเภทแบบหลายคลาส (multiclass classification)
evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")

# คำนวณความแม่นยำ (accuracy) ของโมเดล
# accuracy เก็บค่าความแม่นยำของโมเดล Logistic Regression
accuracy = evaluator.evaluate(predictions)

# แสดงค่าความแม่นยำ (accuracy) ของโมเดล
print(f"Accuracy: {accuracy}")
