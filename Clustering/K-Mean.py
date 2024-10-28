from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import pandas as pd

# สร้าง SparkSession สำหรับการทำงานกับ Spark
# SparkSession คือจุดเริ่มต้นที่ใช้ในการเชื่อมต่อกับคลัสเตอร์ Spark และทำงานต่างๆ ในแอปพลิเคชัน
spark = SparkSession \
    .builder \
    .appName("testKMeans") \
    .getOrCreate()  # สร้างหรือดึง SparkSession ที่มีอยู่แล้วมาใช้งาน
    #appName("testKMeans") \ กำหนดชื่อแอปพลิเคชัน

# อ่านไฟล์ CSV โดยกำหนดว่ามี header (หัวข้อของคอลัมน์) อยู่ในไฟล์
df = spark.read.format("csv").\
    option("header",True).\
    load("fb_live_thailand.csv")  # โหลดข้อมูลจากไฟล์ "fb_live_thailand.csv"

# แปลงข้อมูลในคอลัมน์ "num_sads" และ "num_reactions" ให้เป็นชนิด Double
# เนื่องจาก KMeans ต้องการข้อมูลตัวเลขในการคำนวณ
df = df.select(df.num_sads.cast(DoubleType()), \
               df.num_reactions.cast(DoubleType()))

# VectorAssembler จะรวมคอลัมน์ "num_sads" และ "num_reactions" เข้าด้วยกันในคอลัมน์ "features"
# เพื่อให้โมเดลสามารถใช้ข้อมูลทั้งสองคอลัมน์นี้ในการทำ clustering
vec_assembler = VectorAssembler(inputCols = ["num_sads", \
                                             "num_reactions"], \
                                outputCol = "features")  # กำหนดคอลัมน์ output เป็น "features"

# ทำการ scaling ข้อมูลในคอลัมน์ "features" เพื่อทำให้ข้อมูลทั้งสองคอลัมน์มีขนาดเทียบเคียงกัน
# StandardScaler ช่วยในการทำ normalization เพื่อให้คอลัมน์ต่างๆ ถูกเปรียบเทียบกันอย่างถูกต้อง
scaler = StandardScaler(inputCol="features", \
                        outputCol="scaledFeatures", \
                        withStd=True, \
                        withMean=False)  # ไม่ทำการ scale โดยหาค่าเฉลี่ย
                        #withStd=True, \ ทำการ scale ด้วยค่า standard deviation

# สร้างรายการ k_values เพื่อเก็บค่า silhouette score สำหรับแต่ละค่า k
k_values =[]

# ลูปเพื่อหาค่า k ที่ดีที่สุดในช่วง 2 ถึง 5 
# ค่า k หมายถึงจำนวนกลุ่ม (clusters) ที่เราต้องการให้ KMeans แบ่งข้อมูล
for i in range(2,5):
    # สร้างโมเดล KMeans สำหรับค่า k แต่ละค่าในลูป
    kmeans = KMeans(featuresCol = "scaledFeatures", \
                    predictionCol = "prediction_col", k = i)
    # สร้าง pipeline ที่ประกอบไปด้วยขั้นตอนการรวมฟีเจอร์ (vec_assembler), scaling และการจัดกลุ่ม (KMeans)
    pipeline = Pipeline(stages = [vec_assembler, scaler, kmeans])
    # ฝึกโมเดลด้วยข้อมูลที่เรามี (fit)
    model = pipeline.fit(df)
    # ทำนายผลการจัดกลุ่มด้วยโมเดล
    output = model.transform(df)
    # ประเมินผลลัพธ์ของการจัดกลุ่มด้วย Silhouette Score
    # Silhouette Score เป็นตัววัดคุณภาพของการจัดกลุ่ม ยิ่งค่าสูงแปลว่าการจัดกลุ่มมีประสิทธิภาพมากขึ้น
    evaluator = ClusteringEvaluator(predictionCol = "prediction_col", \
                                    featuresCol = "scaledFeatures", \
                                    metricName = "silhouette", \
                                    distanceMeasure = "squaredEuclidean")
    # คำนวณ Silhouette Score และเก็บค่าไว้ในลิสต์ k_values
    score = evaluator.evaluate(output)
    k_values.append(score)  # เก็บค่า silhouette score ไว้
    print("Silhoutte Score:",score)  # แสดงค่า Silhouette Score สำหรับค่า k ในแต่ละรอบ

# หา k ที่ให้ค่า Silhouette Score ที่ดีที่สุด
# โดยการหาค่า silhouette score ที่สูงที่สุดจากในลิสต์ k_values
best_k = k_values.index(max(k_values)) + 2  # หาค่า k ที่ดีที่สุดจากค่า silhouette score
print("The best k", best_k, max(k_values))  # แสดงค่า k ที่ดีที่สุดและค่า Silhouette Score ที่สูงที่สุด

# สร้างโมเดล KMeans ใหม่ โดยใช้ค่า k ที่ดีที่สุดที่ได้จากการประเมิน
kmeans = KMeans(featuresCol = "scaledFeatures", \
                predictionCol = "prediction_col", \
                k = best_k)

# สร้าง pipeline ใหม่อีกครั้งรวมถึงขั้นตอนการรวมฟีเจอร์, scaling และ KMeans clustering
pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])

# ฝึกโมเดลด้วยข้อมูลเดิมและค่า k ที่ดีที่สุด (fit)
model = pipeline.fit(df)

# ทำนายผลลัพธ์ของการจัดกลุ่มโดยใช้โมเดลที่ฝึกใหม่
predictions = model.transform(df)

# ประเมินผลลัพธ์การจัดกลุ่มอีกครั้งด้วย Silhouette Score
evaluator = ClusteringEvaluator(predictionCol = "prediction_col", \
                                featuresCol = "scaledFeatures", \
                                metricName = "silhouette", \
                                distanceMeasure = "squaredEuclidean")
# คำนวณค่า Silhouette Score สำหรับผลการจัดกลุ่มใหม่
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " \
      + str(silhouette))  # แสดงค่า Silhouette Score หลังจากทำนายผล

# แปลงข้อมูลจาก Spark DataFrame เป็น Pandas DataFrame เพื่อใช้ในการแสดงผลด้วย matplotlib
clustered_data_pd = predictions.toPandas()

# แสดงผลการจัดกลุ่มด้วย scatter plot
# โดยแต่ละจุดจะแสดงเป็นสีต่างๆ ตามคลัสเตอร์ที่โมเดลจัดกลุ่มให้
plt.scatter(clustered_data_pd["num_reactions"], \
            clustered_data_pd["num_sads"], \
            c = clustered_data_pd["prediction_col"])  # ใช้สีแสดงคลัสเตอร์ของแต่ละจุด
plt.xlabel("num_reactions")  # ป้ายแกน x เป็นจำนวนปฏิกิริยา (num_reactions)
plt.ylabel("num_sads")  # ป้ายแกน y เป็นจำนวนอิโมจิเศร้า (num_sads)
plt.title("K-means Clustering")  # ตั้งชื่อกราฟว่า K-means Clustering
plt.colorbar().set_label("Cluster")  # เพิ่มแถบสีเพื่อแสดงว่าแต่ละสีหมายถึงกลุ่มไหน
plt.show()  # แสดงกราฟผลลัพธ์การจัดกลุ่ม
