from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# เริ่มต้นสร้าง SparkSession ซึ่งเป็นตัวจัดการสภาพแวดล้อม Spark
spark = SparkSession.builder \
    .appName("BookRecommendationALS") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV ซึ่งเป็นข้อมูลการให้คะแนนหนังสือ
# กำหนดให้มีการอ่านแถวแรกเป็น header และให้ Spark ทำนาย schema ของคอลัมน์โดยอัตโนมัติ
data = spark.read.csv('book_ratings.csv', header=True, inferSchema=True)

# แสดง schema ของข้อมูลเพื่อให้เห็นโครงสร้างของข้อมูล
data.printSchema()

# แสดงตัวอย่างข้อมูล 5 แถวแรก
data.show(5)

# กำหนดค่า ALS (Alternating Least Squares) model สำหรับการแนะนำ
als = ALS(
    maxIter=10,  # จำนวน iteration ที่จะทำซ้ำ (ยิ่งมากยิ่งแม่น แต่ใช้เวลานานขึ้น)
    userCol="user_id",  # คอลัมน์สำหรับ user ID
    itemCol="book_id",  # คอลัมน์สำหรับ book ID
    ratingCol="rating",  # คอลัมน์ที่เก็บค่าการให้คะแนน
    coldStartStrategy="drop"  # จัดการข้อมูลที่มีการทำนายเป็น NaN โดยลบข้อมูลนั้นทิ้ง
)

# ฝึกโมเดล ALS โดยใช้ข้อมูลที่เราอ่านเข้ามา
model = als.fit(data)

# ทำนายการให้คะแนน โดยใช้โมเดลที่ฝึกมาแล้ว
predictions = model.transform(data)

# กำหนดตัวประเมินผลแบบ RegressionEvaluator เพื่อตรวจสอบคุณภาพของโมเดล
# ในที่นี้เราจะใช้ Root Mean Squared Error (RMSE)
evaluator = RegressionEvaluator(
    metricName="rmse",  # ตัวชี้วัดเป็น RMSE
    labelCol="rating",  # คอลัมน์ที่ใช้เป็นค่าจริง
    predictionCol="prediction"  # คอลัมน์ที่ใช้เป็นค่าทำนายจากโมเดล
)

# ประเมินโมเดลโดยคำนวณค่า RMSE จากการทำนาย
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# แสดงผลการทำนายเฉพาะของผู้ใช้ที่มี user_id เท่ากับ 53
user_id = 53 # user_ids = [53, 54, 55] มากกว่า 2 คน
user_predictions = predictions.filter(col("user_id") == user_id)
user_predictions = user_predictions.select("book_id", "user_id", "rating", "prediction").orderBy(col("prediction").desc()) 
# desc (descending order) มากไปน้อย -  asc() ascending order น้อยไปมาก

user_predictions.show(truncate=False)  # แสดงผลการทำนายทั้งหมดโดยไม่ตัดทอน

# แสดง 5 หนังสือที่โมเดลแนะนำให้สำหรับผู้ใช้ทั้งหมด recommendForAllUsers(5) เติมแล้วแต่่จำนวนกำหนดได้เลย
user_recommendations = model.recommendForAllUsers(5)
user_recommendations.show(truncate=False)

# แสดงผู้ใช้ที่แนะนำ 5 รายสำหรับหนังสือทุกเล่ม recommendForAllItems(5) เติมแล้วแต่่จำนวนกำหนดได้เลย
item_recommendations = model.recommendForAllItems(5)
item_recommendations.show(truncate=False)

# หยุดการทำงานของ SparkSession เมื่อเสร็จสิ้น
spark.stop()
