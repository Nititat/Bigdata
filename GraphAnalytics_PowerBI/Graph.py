from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("AirlineRoutesGraph") \
    .config("spark.sql.warehouse.dir", "file:///C:/temp") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ CSV
airline_routes_df = spark.read.csv(r"C:\Users\rawip\Downloads\BigData\GraphAnalytics_PowerBI\airline_routes.csv", header=True, inferSchema=True)

# สร้าง DataFrame สำหรับ vertices และ edges
vertices = airline_routes_df.select("source_airport").withColumnRenamed("source_airport", "id").distinct()
edges = airline_routes_df.select("source_airport", "destination_airport") \
    .withColumnRenamed("source_airport", "src") \
    .withColumnRenamed("destination_airport", "dst")

# สร้าง GraphFrame โดยใช้ vertices และ edges
graph = GraphFrame(vertices, edges)

# แสดงจำนวน vertices และ edges
print("Number of vertices:", graph.vertices.count())
print("Number of edges:", graph.edges.count())

# จัดกลุ่ม edges และเพิ่มคอลัมน์สี
grouped_edges = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .orderBy(desc("count")) \
    .withColumn("source_color", lit("#3358FF")) \
    .withColumn("destination_color", lit("#FF3F33"))

# เขียนข้อมูลเป็นไฟล์ CSV
grouped_edges.coalesce(1).write.csv("grouped_airline_routes1.csv", mode="overwrite", header=True)
