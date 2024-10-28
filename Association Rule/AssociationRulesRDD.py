from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import collect_list, array_distinct, explode, split, col

# ขั้นตอนที่ 1: สร้าง SparkSession
spark = SparkSession.builder.appName("FPGrowthExample").getOrCreate()

# ขั้นตอนที่ 2: อ่านข้อมูลจากไฟล์ CSV
data = spark.read.csv("groceries_data.csv", header=True, inferSchema=True)

# ขั้นตอนที่ 3: รวมกลุ่มข้อมูลตามหมายเลขสมาชิกและรวบรวมรายการสินค้า
grouped_data = data.groupBy("Member_number").agg(collect_list("itemDescription").alias("Items"))

# ขั้นตอนที่ 4: แสดงข้อมูลที่รวมกลุ่มแล้ว
grouped_data.show(truncate=False)

# ขั้นตอนที่ 5: เพิ่มคอลัมน์ 'basket' ที่มีรายการสินค้าที่ไม่ซ้ำกันในแต่ละตะกร้า
grouped_data = grouped_data.withColumn("basket", array_distinct(grouped_data["Items"]))

# ขั้นตอนที่ 6: แสดงข้อมูลที่อัปเดตแล้ว
grouped_data.show(truncate=False)

# ขั้นตอนที่ 7: สร้างโมเดล FPGrowth ด้วยพารามิเตอร์ที่กำหนด
minSupport = 0.1
minConfidence = 0.2
fp = FPGrowth(minSupport=minSupport, minConfidence=minConfidence, itemsCol='basket', predictionCol='prediction')

# ขั้นตอนที่ 8: ฝึกโมเดล FPGrowth
model = fp.fit(grouped_data)

# ขั้นตอนที่ 9: แสดง itemsets ที่พบบ่อย
model.freqItemsets.show(10)

# ขั้นตอนที่ 10: กรองกฎความสัมพันธ์ตามค่า confidence
filtered_rules = model.associationRules.filter(model.associationRules.confidence > 0.4)

# ขั้นตอนที่ 11: แสดงกฎความสัมพันธ์ที่กรองแล้ว
filtered_rules.show(truncate=False)

# ขั้นตอนที่ 12: สร้าง DataFrame ใหม่ที่มีตะกร้าสำหรับการทำนาย
new_data_rdd = spark.sparkContext.parallelize([
    (['vegetable juice', 'frozen fruits', 'packaged fruit'],),
    (['mayonnaise', 'butter', 'buns'],)
])

# ขั้นตอนที่ 13: สร้าง DataFrame จาก RDD
new_data = spark.createDataFrame(new_data_rdd, ["basket"])

# ขั้นตอนที่ 14: แสดงข้อมูลใหม่สำหรับการทำนาย
new_data.show(truncate=False)

# ขั้นตอนที่ 15: ใช้โมเดลในการทำนาย
predictions = model.transform(new_data)

# ขั้นตอนที่ 16: แสดงผลการทำนาย
predictions.show(truncate=False)

# ปิด SparkSession
spark.stop()
