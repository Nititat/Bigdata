#โค้ดนี้ใช้ PySpark เพื่อสร้างและประเมินโมเดล Decision Tree Regression สำหรับการพยากรณ์ โดยใช้ข้อมูลจากไฟล์ fb_live_thailand.csv ที่มีคอลัมน์ num_reactions และ num_loves ซึ่งอธิบายการทำงานของ
# ปรับให้ดี ค่า ใกล้ 1

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# สร้าง SparkSession
spark = SparkSession.builder.appName("DecisionTreeRegression").getOrCreate()

# โหลดชุดข้อมูลจากไฟล์ CSV
df = spark.read.format("csv").option("header", True).load("fb_live_thailand.csv")

# แปลงคอลัมน์ num_reactions และ num_loves เป็นชนิดข้อมูล Double
df = df.select(df.num_reactions.cast("Double"), df.num_loves.cast("Double")) # Float 32 บิต Double 64 บิต

# ลบแถวที่มีค่า null ค่าว่าง (ถ้าnot null ไม่ว่าง)
df = df.na.drop()
# df = df.na.drop()               # ลบแถวที่มีค่า null ทั้งหมด
# df = df.na.fill(0)              # แทนค่าที่เป็น null ด้วย 0 ในทุกคอลัมน์
# df = df.na.fill({"age": 18})    # แทนค่าที่เป็น null ในคอลัมน์ "age" ด้วย 18


# ใช้ StringIndexer เพื่อแปลงคอลัมน์ num_reactions และ num_loves เป็นค่าดัชนี
indexer_reactions = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")
indexer_loves = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

# ใช้ OneHotEncoder เพื่อแปลงค่าดัชนีเป็นเวกเตอร์ที่เข้ารหัสแบบ One-Hot
encoder_reactions = OneHotEncoder(inputCols=["num_reactions_ind"], outputCols=["reactions_encoded"])
encoder_loves = OneHotEncoder(inputCols=["num_loves_ind"], outputCols=["loves_encoded"])

# ใช้ VectorAssembler เพื่อรวมฟีเจอร์เข้าด้วยกันเป็นเวกเตอร์เดียว
vec_assembler = VectorAssembler(inputCols=["reactions_encoded", "loves_encoded"], outputCol="features")

# สร้าง Pipeline ที่ประกอบด้วยขั้นตอนต่าง ๆ
pipeline = Pipeline(stages=[indexer_reactions, indexer_loves, encoder_reactions, encoder_loves, vec_assembler])

# ฝึก Pipeline model โดยใช้ชุดข้อมูล
pipeline_model = pipeline.fit(df)

# ทำการแปลงข้อมูลด้วย Pipeline model
df_transformed = pipeline_model.transform(df)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=1234)

# สร้างโมเดล Decision Tree Regressor พร้อมตั้งค่า hyperparameters
dt_regressor = DecisionTreeRegressor(featuresCol="features", labelCol="num_loves_ind", maxDepth=30, minInstancesPerNode=2) # tune Model  maxDepth=30, minInstancesPerNode=2 ปรับตรงนี้

# ฝึกโมเดลด้วยชุดฝึก
dt_model = dt_regressor.fit(train_df)

# ทำการทำนายด้วยโมเดลที่ฝึกมา
predictions = dt_model.transform(test_df)

# สร้างตัวประเมินผลเพื่อประเมินโมเดล
evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

# คำนวณคะแนน R2
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print()
print('='*50)
print()
print(f"R2: {r2}")
print()
print('='*50)
print()

spark.stop()