from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType

# สร้าง SparkSession เพื่อเริ่มการทำงานกับ Spark
spark = SparkSession.builder \
    .appName("Linear Regression Analysis") \
    .getOrCreate()  # สร้างหรือดึง SparkSession มาใช้งาน
    #appName("Linear Regression Analysis") \  # กำหนดชื่อแอปพลิเคชัน

# โหลดไฟล์ CSV ลงใน DataFrame พร้อมกำหนดให้มีการตรวจสอบชนิดของข้อมูลโดยอัตโนมัติ (inferSchema=True)
data = spark.read.csv('fb_live_thailand.csv', header=True, inferSchema=True)

# แสดงข้อมูลบางส่วนของ DataFrame ที่โหลดมา เพื่อดูโครงสร้างข้อมูล
data.show()

# ใช้ VectorAssembler ในการรวมคอลัมน์ 'num_reactions' และ 'num_loves' ให้เป็นฟีเจอร์ที่ชื่อว่า 'features'
assembler = VectorAssembler(
    inputCols=['num_reactions', 'num_loves'],  # คอลัมน์ที่ต้องการรวมเป็นฟีเจอร์
    outputCol='features'  # ชื่อฟีเจอร์ที่สร้างขึ้นใหม่
)

# แปลงข้อมูลด้วย VectorAssembler โดยรวมคอลัมน์ตามที่กำหนดไว้และสร้างคอลัมน์ 'features'
data_assembled = assembler.transform(data)

# แสดงข้อมูลหลังจากที่แปลงเสร็จแล้ว (มีคอลัมน์ 'features' เพิ่มขึ้น)
data_assembled.show()

# สร้างโมเดล Linear Regression
linear_regression = LinearRegression(
    labelCol='num_loves',  # คอลัมน์ที่เป็น label (ค่าที่ต้องการทำนาย)
    featuresCol='features',  # คอลัมน์ฟีเจอร์ (ข้อมูลที่ใช้ในการทำนาย)
    maxIter=10,  # กำหนดจำนวนรอบการทำซ้ำ (iter) สูงสุด
    regParam=0.3,  # ค่า Regularization parameter (เพื่อป้องกัน overfitting)
    elasticNetParam=0.8  # ค่า ElasticNet mixing parameter (การผสมผสานระหว่าง L1 และ L2)
)

# สร้าง pipeline ที่ประกอบด้วยขั้นตอนการทำ linear regression
pipeline = Pipeline(stages=[linear_regression])

# แบ่งข้อมูลออกเป็นชุดการฝึก (train_data) และชุดการทดสอบ (test_data) โดยแบ่งเป็น 80% และ 20%
train_data, test_data = data_assembled.randomSplit([0.8, 0.2], seed=42)

# ฝึกโมเดลด้วยข้อมูล train_data โดยใช้ pipeline ที่เราสร้าง
pipeline_model = pipeline.fit(train_data)

# ใช้โมเดลที่ฝึกเสร็จแล้วในการทำนายข้อมูล test_data
predictions = pipeline_model.transform(test_data)

# แสดง 5 แถวของ DataFrame ที่มีการทำนายผลแล้ว
predictions.select('num_loves', 'features', 'prediction').show(5)

# สร้าง RegressionEvaluator เพื่อประเมินผลลัพธ์ของโมเดล
evaluator = RegressionEvaluator(
    labelCol='num_loves',  # คอลัมน์ที่เป็นค่าจริง (label)
    predictionCol='prediction'  # คอลัมน์ที่เป็นค่าทำนาย
)

# คำนวณค่า Mean Squared Error (MSE) เพื่อประเมินความถูกต้องของโมเดล
mse = evaluator.setMetricName("mse").evaluate(predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")  # แสดงค่า MSE

# คำนวณค่า R2 ซึ่งเป็นการวัดความเหมาะสมของโมเดล (ใกล้ 1 แปลว่าโมเดลดี)
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2: {r2:.4f}")  # แสดงค่า R2

# แปลงข้อมูลจาก Spark DataFrame เป็น Pandas DataFrame เพื่อใช้ในการสร้างกราฟ
pandas_df = predictions.select('num_loves', 'prediction').toPandas()

# สร้าง scatter plot โดยใช้ seaborn เพื่อแสดงการกระจายตัวของข้อมูลระหว่างค่าจริง (num_loves) และค่าทำนาย (prediction)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_loves', y='prediction', data=pandas_df)
plt.title('Scatter Plot of num_loves vs Prediction')  # ตั้งชื่อกราฟ
plt.xlabel('num_loves')  # ตั้งชื่อแกน X
plt.ylabel('Prediction')  # ตั้งชื่อแกน Y
plt.show()  # แสดงกราฟ scatter plot

# เลือกเฉพาะคอลัมน์ num_loves และ prediction
# ทำการแปลงข้อมูลเหล่านี้ให้เป็น IntegerType และเรียงลำดับข้อมูลตาม prediction จากมากไปน้อย
selected_data = predictions.select(
    col('num_loves').cast(IntegerType()).alias('num_loves'),  # แปลงคอลัมน์ num_loves ให้เป็น Integer
    col('prediction').cast(IntegerType()).alias('prediction')  # แปลงคอลัมน์ prediction ให้เป็น Integer
).orderBy(col('prediction').desc())  # เรียงลำดับจากมากไปน้อย

# แปลงข้อมูล selected_data ให้เป็น Pandas DataFrame เพื่อใช้ในการสร้างกราฟ
pandas_df = selected_data.toPandas()

# สร้างกราฟเชิงเส้น (linear regression plot) โดยใช้ seaborn เพื่อแสดงความสัมพันธ์ระหว่าง num_loves และ prediction
plt.figure(figsize=(10, 6))
sns.lmplot(x='num_loves', y='prediction', data=pandas_df, aspect=1.5)

# แสดงกราฟ linear regression plot
plt.title('Linear Regression: num_loves vs Prediction')  # ตั้งชื่อกราฟ
plt.xlabel('num_loves')  # ตั้งชื่อแกน X
plt.ylabel('Prediction')  # ตั้งชื่อแกน Y
plt.show()  # แสดงกราฟผลลัพธ์
