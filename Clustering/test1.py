from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

# สร้าง SparkSession สำหรับการทำงานกับ Spark
spark = SparkSession.builder.appName("KMeansClusteringExample").getOrCreate()

# อ่านข้อมูล DataFrame (สมมติว่า DataFrame มีคอลัมน์ num_likes, num_comments, num_shares)
df = spark.read.format("csv").option("header", True).load("fb_live_thailand.csv")

# แปลงข้อมูลในคอลัมน์ num_likes, num_comments, num_shares ให้เป็นชนิด Double
df = df.select(df.num_likes.cast("double"), df.num_comments.cast("double"), df.num_shares.cast("double"))

# รวมคอลัมน์ num_likes, num_comments, และ num_shares เป็นฟีเจอร์เดียวที่ชื่อว่า "features"
vec_assembler = VectorAssembler(inputCols=["num_likes", "num_comments", "num_shares"], outputCol="features")

# ทำการ scaling ข้อมูลในคอลัมน์ "features"
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# สร้างลิสต์เพื่อเก็บค่า Silhouette Score และค่า k ที่ทดสอบ
k_values = []
silhouette_scores = []

# ทดสอบค่า k ตั้งแต่ 3 ถึง 6
for k in range(3, 7):
    # สร้างโมเดล KMeans สำหรับค่า k ที่ทดสอบ
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=k)

    # สร้าง Pipeline ที่ประกอบด้วยการรวมฟีเจอร์และการจัดกลุ่มด้วย KMeans
    pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans])

    # ฝึกโมเดลด้วยข้อมูลที่มี
    model = pipeline.fit(df)

    # ทำนายผลลัพธ์การจัดกลุ่ม
    predictions = model.transform(df)

    # ประเมินผลลัพธ์การจัดกลุ่มด้วย Silhouette Score
    evaluator = ClusteringEvaluator(featuresCol="scaledFeatures", predictionCol="prediction", metricName="silhouette", distanceMeasure="squaredEuclidean")
    
    # คำนวณค่า Silhouette Score สำหรับแต่ละค่า k และเก็บค่าไว้ในลิสต์
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)
    k_values.append(k)
    
    # แสดงค่า k และ Silhouette Score
    print(f"Silhouette Score for k={k}: {silhouette}")

# ปิด SparkSession หลังจากใช้งาน
spark.stop()

# สร้างกราฟแสดงค่า Silhouette Score สำหรับค่า k ต่างๆ
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k Values in K-means Clustering')
plt.grid(True)
plt.show()
