from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, desc
import seaborn as sns
import matplotlib.pyplot as plt

#Linear

# Linear Regression Pipeline Example
# สร้าง SparkSession เพื่อใช้ในการทำงานกับ PySpark
spark = SparkSession.builder \
    .appName("LinearRegressionPipelineExample") \
    .getOrCreate()

# ขั้นตอนที่ 2: โหลดข้อมูลจากไฟล์ CSV ลงใน DataFrame
data = spark.read.csv('fb_live_thailand.csv', header=True, inferSchema=True)

# ขั้นตอนที่ 3: ใช้ StringIndexer เพื่อเตรียมข้อมูล
# แปลงคอลัมน์ num_reactions เป็นค่าที่เป็นตัวเลข
indexer_reactions = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")
# แปลงคอลัมน์ num_loves เป็นค่าที่เป็นตัวเลข
indexer_loves = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

# ขั้นตอนที่ 4: ใช้ VectorAssembler เพื่อสร้างเวกเตอร์ฟีเจอร์
assembler = VectorAssembler(
    inputCols=["num_reactions_ind", "num_loves_ind"],
    outputCol="features"
)

# ปรับค่า โมเดล
# ขั้นตอนที่ 5: กำหนดโมเดล Linear Regression
lr = LinearRegression(
    featuresCol="features",
    labelCol="num_loves_ind",
    maxIter=10,         # จำนวนรอบการฝึกฝน 10 รอบ
    regParam=0.1,       # พารามิเตอร์การลดความซับซ้อน
    elasticNetParam=0.8 # ค่าที่ใช้ในการควบคุมระหว่าง L1 และ L2 regularization
)

# ค่า ElasticNetParam สามารถกำหนดได้ระหว่าง 0 ถึง 1:
# หากกำหนดค่าเป็น 0 โมเดลจะใช้เฉพาะ L2 Regularization (Ridge) ** เอา Columns มาทั้งหมดแต่ไม่ให้ความสำคัญ
# หากกำหนดค่าเป็น 1 โมเดลจะใช้เฉพาะ L1 Regularization (Lasso) ** ตัด Columns บ้างส่วนออกไปเลยที่ไม่สำคัญ
# หากกำหนดค่า ระหว่าง 0 และ 1 โมเดลจะใช้การผสมผสานระหว่าง L1 และ L2 , 0.5 0.8 0.7 ประมาณนี้

# ขั้นตอนที่ 6: สร้าง Pipeline ด้วยแต่ละขั้นตอน
# ไม่รวม indexer_reactions และ indexer_loves ใน pipeline
pipeline = Pipeline(stages=[assembler, lr]) # stages=[assembler, lr] ทำงานตามลำดับขั้นตอน
                                            # assembler --> lr

# ขั้นตอนที่ 7: แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234) # randomSplit ถ้าไม่มีจะเกิด Overfitting

# ขั้นตอนที่ 8: ฝึกโมเดลด้วยชุดฝึก
# ใช้ StringIndexers เพื่อแปลงข้อมูลในชุดฝึก
train_data = indexer_reactions.fit(train_data).transform(train_data)
train_data = indexer_loves.fit(train_data).transform(train_data)
pipeline_model = pipeline.fit(train_data)

# ขั้นตอนที่ 9: แปลงข้อมูลชุดทดสอบด้วยโมเดล pipeline
# ใช้ StringIndexers เพื่อแปลงข้อมูลในชุดทดสอบ
test_data = indexer_reactions.fit(test_data).transform(test_data)
test_data = indexer_loves.fit(test_data).transform(test_data)
predictions = pipeline_model.transform(test_data)

# ขั้นตอนที่ 10: สร้าง RegressionEvaluator
evaluator = RegressionEvaluator(
    labelCol="num_loves_ind",
    predictionCol="prediction"
)

print()
print('='*50)
print()

# MSE ใกล้เคียงกับ 0 ดีสุด
# ประเมิน Mean Squared Error (MSE)
evaluator.setMetricName("mse")
mse = evaluator.evaluate(predictions)
print(f"Mean Squared Error (MSE): {mse}")

# R2 ใกล้เคียงกับ 1 ดีสุด
# ประเมิน R-squared (R2)
evaluator.setMetricName("r2")
r2 = evaluator.evaluate(predictions)
print(f"R-squared (R2): {r2}")
print()
print('='*50)
print()

# ขั้นตอนที่ 11: เลือกคอลัมน์และแปลงประเภท, เรียงลำดับตาม 'prediction' ในลำดับลดลง
# เลือกเฉพาะคอลัมน์ num_loves_ind และ prediction พร้อมแปลงเป็นประเภท Integer 
# และเรียงข้อมูลตาม prediction ในลำดับลดลง

selected_data = predictions.select(
    col("num_loves_ind").cast(IntegerType()).alias("num_loves"),
    col("prediction").cast(IntegerType()).alias("prediction")
).orderBy(col("prediction").desc())

# แปลงเป็น Pandas DataFrame สำหรับการวิเคราะห์เพิ่มเติม
selected_data_pd = selected_data.toPandas()
print(selected_data_pd)

# ขั้นตอนที่ 12: การสร้างกราฟโดยใช้ Seaborn's lmplot
plt.figure(figsize=(12, 6))
sns.regplot(
    data=selected_data_pd,
    x='num_loves',
    y='prediction',
    scatter_kws={'s': 50, 'alpha': 0.5},  # กำหนดคุณสมบัติของจุดในกราฟ
    line_kws={'color': 'red'}  # กำหนดคุณสมบัติของเส้นในกราฟ
)

plt.title('Linear Regression: Actual vs Predicted Values') # ชื่อกราฟ
plt.xlabel('Actual num_loves') # ชื่อแกน x
plt.ylabel('Predicted num_loves') # ชื่อแกน y
plt.show() # แสดงกราฟ

# หยุด SparkSession
spark.stop()