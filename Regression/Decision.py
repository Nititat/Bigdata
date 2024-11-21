# 1. Import libraries
# นำเข้าไลบรารีที่จำเป็นจาก PySpark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# สร้าง SparkSession ซึ่งเป็นจุดเริ่มต้นสำหรับการทำงานกับ PySpark
spark = SparkSession.builder \
    .appName("DecisionTreeRegressionExample") \
    .getOrCreate()  # สร้าง SparkSession

# โหลดข้อมูล CSV ลงใน DataFrame พร้อม inferSchema เพื่อให้ตรวจสอบชนิดข้อมูลโดยอัตโนมัติ
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# StringIndexer ใช้ในการแปลงคอลัมน์ที่เป็นตัวอักษรให้เป็นตัวเลข (ถ้าคอลัมน์เป็นตัวอักษร)
# ในที่นี้แปลงคอลัมน์ 'num_reactions' และ 'num_loves' เป็นเลข ID
indexer_reactions = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")
indexer_loves = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

# OneHotEncoder ใช้ในการแปลงข้อมูลตัวเลขที่ได้จาก StringIndexer ให้เป็นเวกเตอร์ที่สามารถใช้ในโมเดลได้
# แปลงคอลัมน์ที่ผ่านการ indexing ('num_reactions_ind' และ 'num_loves_ind') ให้เป็นเวกเตอร์
encoder_reactions = OneHotEncoder(inputCols=["num_reactions_ind"], outputCols=["num_reactions_vec"])
encoder_loves = OneHotEncoder(inputCols=["num_loves_ind"], outputCols=["num_loves_vec"])

# VectorAssembler ใช้ในการรวมฟีเจอร์หลายคอลัมน์ (ในที่นี้คือเวกเตอร์ที่ได้จากการ encoding) เป็นคอลัมน์เดียวที่ชื่อว่า 'features'
assembler = VectorAssembler(inputCols=["num_reactions_vec", "num_loves_vec"], outputCol="features")

# Pipeline ใช้ในการรวมขั้นตอนต่างๆ ตั้งแต่การทำ indexing, encoding, จนถึงการรวมฟีเจอร์
pipeline = Pipeline(stages=[indexer_reactions, indexer_loves, encoder_reactions, encoder_loves, assembler])

# Fit ข้อมูลใน pipeline เพื่อทำให้ข้อมูลผ่านขั้นตอนการแปลงตามที่กำหนดไว้
pipeline_model = pipeline.fit(data)

# Transform ข้อมูลให้ผ่าน pipeline ที่สร้างขึ้น
transformed_data = pipeline_model.transform(data)

# แบ่งข้อมูลออกเป็นชุดฝึก (train_data) และชุดทดสอบ (test_data) โดยแบ่งเป็น 80% สำหรับฝึก และ 20% สำหรับทดสอบ
train_data, test_data = transformed_data.randomSplit([0.8, 0.2])

# สร้างโมเดล DecisionTreeRegressor ซึ่งเป็นโมเดลการถดถอยโดยใช้ decision tree
# labelCol คือคอลัมน์ที่ใช้เป็นค่าจริง (target) และ featuresCol คือฟีเจอร์ที่ใช้ในการทำนาย
dt = DecisionTreeRegressor(labelCol="num_loves_ind", featuresCol="features")

# ฝึกโมเดล DecisionTreeRegressor ด้วยข้อมูลฝึก
dt_model = dt.fit(train_data)

# ใช้โมเดลที่ฝึกเสร็จแล้วเพื่อทำนายข้อมูลทดสอบ
predictions = dt_model.transform(test_data)

# สร้าง RegressionEvaluator เพื่อใช้ประเมินผลลัพธ์ของการทำนาย
# labelCol คือค่าจริง (target) และ predictionCol คือค่าที่โมเดลทำนาย
evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

# ประเมินผลลัพธ์โดยการคำนวณค่า R2 ซึ่งเป็นตัวชี้วัดคุณภาพของโมเดล (ค่า R2 ใกล้ 1 แปลว่าโมเดลทำนายได้ดี)
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2 score: {r2}")  # แสดงผลค่า R2

# หยุดการทำงานของ SparkSession หลังจากใช้งานเสร็จสิ้น
spark.stop()
