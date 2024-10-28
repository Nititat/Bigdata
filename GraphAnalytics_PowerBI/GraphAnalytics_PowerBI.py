from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("AirlineRoutesGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ airline_routes.csv
airline_routes_df = spark.read.csv("C:/Users/rawip/Downloads/BigData/GraphAnalytics_PowerBI/airline_routes.csv", header=True, inferSchema=True)

# แสดง DataFrame
airline_routes_df.show()

# สร้าง DataFrame สำหรับ vertices โดยใช้ withColumnRenamed() และ source_airport เป็น id
vertices = airline_routes_df.select("source_airport").withColumnRenamed("source_airport", "id").distinct()

# สร้าง DataFrame สำหรับ edges โดยใช้ withColumnRenamed() โดยให้ source_airport เป็น src และ destination_airport เป็น dst
edges = airline_routes_df.select("source_airport", "destination_airport") \
    .withColumnRenamed("source_airport", "src") \
    .withColumnRenamed("destination_airport", "dst")

# แสดง DataFrame สำหรับ vertices
vertices.show()

# แสดง DataFrame สำหรับ edges
edges.show()

# สร้าง GraphFrame โดยใช้ vertices และ edges ที่สร้างขึ้น
graph = GraphFrame(vertices, edges)


# แสดงจำนวน vertices
print("Number of vertices:", graph.vertices.count())

# แสดงจำนวน edges
print("Number of edges:", graph.edges.count())

# กลุ่ม edges โดยใช้ src และ dst, กรองตาม count > 5, เพิ่มคอลัมน์ source_color และ destination_color
grouped_edges = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .orderBy(desc("count")) \
    .withColumn("source_color", lit("#3358FF")) \
    .withColumn("destination_color", lit("#FF3F33"))

# แสดงข้อมูลที่ถูกจัดกลุ่ม
grouped_edges.show()

# เขียนข้อมูลที่ถูกจัดกลุ่มลงในไฟล์ CSV โดยใช้โหมด overwrite และตั้ง header เป็น True
grouped_edges.write.csv("grouped_airline_routes.csv", mode="overwrite", header=True)

print('='*80)
print('')
