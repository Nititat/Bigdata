# นำเข้าไลบรารีที่จำเป็น
from pyspark.sql import SparkSession  # ใช้สำหรับการสร้าง SparkSession
from graphframes import GraphFrame  # ใช้สำหรับสร้างกราฟด้วย GraphFrame
from pyspark.sql.functions import desc, col  # ใช้สำหรับการจัดเรียงข้อมูลและการเลือกคอลัมน์

# สร้าง SparkSession สำหรับการทำงานกับ Spark และ GraphFrame
spark = SparkSession.builder \
    .appName("Graph Analytics") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()  # สร้าง SparkSession

# สร้าง DataFrame ของ vertices (โหนดในกราฟ)
# vertices: ตัวแปรที่เก็บข้อมูลของโหนด (บุคคล) ซึ่งมีสองคอลัมน์คือ id (ชื่อ) และ age (อายุ)
vertices = spark.createDataFrame([
    ("Alice", 45),
    ("Jacob", 43),
    ("Roy", 21),
    ("Ryan", 49),
    ("Emily", 24),
    ("Sheldon", 52)
], ["id", "age"])

# สร้าง DataFrame ของ edges (ความสัมพันธ์ระหว่างโหนด)
# edges: ตัวแปรที่เก็บข้อมูลความสัมพันธ์ระหว่างโหนด มีสามคอลัมน์คือ src (ต้นทาง), dst (ปลายทาง), และ relation (ความสัมพันธ์)
edges = spark.createDataFrame([
    ("Sheldon", "Alice", "Sister"),
    ("Alice", "Jacob", "Husband"),
    ("Emily", "Jacob", "Father"),
    ("Ryan", "Alice", "Friend"),
    ("Alice", "Emily", "Daughter"),
    ("Alice", "Roy", "Son"),
    ("Jacob", "Roy", "Son")
], ["src", "dst", "relation"])

# สร้างกราฟจาก DataFrame ของ vertices และ edges
# graph: ตัวแปรที่เก็บกราฟซึ่งสร้างจาก vertices และ edges
graph = GraphFrame(vertices, edges)

# แสดงข้อมูลของโหนดทั้งหมดในกราฟ
print("Vertices:")
graph.vertices.show()

# แสดงข้อมูลของความสัมพันธ์ทั้งหมดในกราฟ
print("Edges:")
graph.edges.show()

# ตัวอย่าง: คำสั่ง groupBy และ orderBy เพื่อจัดกลุ่มและเรียงลำดับตามจำนวนความสัมพันธ์ (edges)
print("Grouped and Ordered Edges by Count:")
graph.edges.groupBy("src", "dst").count().orderBy(desc("count")).show()

# ตัวอย่าง: กรองข้อมูลของความสัมพันธ์ที่ src (ต้นทาง) หรือ dst (ปลายทาง) เป็น "Alice"
print("Filtered Edges where src = 'Alice' or dst = 'Alice':")
filtered_edges = graph.edges.where("src = 'Alice' OR dst = 'Alice'")
filtered_edges.show()

# สร้าง subgraph ใหม่จากการกรอง edges ที่ src หรือ dst เป็น "Alice"
# subgraph: ตัวแปรที่เก็บกราฟย่อยจากกราฟหลัก โดยมีเพียง edges ที่ผ่านการกรอง
print("Subgraph with filtered edges:")
subgraph = GraphFrame(graph.vertices, filtered_edges)
subgraph.edges.show()

# ตัวอย่าง: Motif finding คือการค้นหารูปแบบในกราฟ (เช่น รูปแบบของโหนดและความสัมพันธ์)
print("Motif finding example:")
motifs = graph.find("(a)-[ab]->(b)")  # หารูปแบบความสัมพันธ์ระหว่าง a และ b
motifs.show()

# ตัวอย่าง: การคำนวณ PageRank ซึ่งเป็นอัลกอริธึมที่ใช้วัดความสำคัญของโหนดในกราฟ
print("PageRank results:")
pagerank = graph.pageRank(resetProbability=0.15, maxIter=5)  # คำนวณ PageRank โดยกำหนด maxIter เป็น 5
pagerank.vertices.orderBy(desc("pagerank")).show()  # แสดงผลการจัดอันดับของโหนดตามค่า PageRank

# ตัวอย่าง: การคำนวณ in-degree และ out-degree ของโหนดในกราฟ
print("In-Degree:")
graph.inDegrees.orderBy(desc("inDegree")).show()  # คำนวณ in-degree ซึ่งคือจำนวนความสัมพันธ์ที่โหนดนั้นเป็นปลายทาง

print("Out-Degree:")
graph.outDegrees.orderBy(desc("outDegree")).show()  # คำนวณ out-degree ซึ่งคือจำนวนความสัมพันธ์ที่โหนดนั้นเป็นต้นทาง

# ตัวอย่าง: การทำ Breadth-First Search (BFS) เพื่อค้นหาเส้นทางจากโหนดหนึ่งไปยังอีกโหนดหนึ่ง
print("Breadth-First Search (BFS):")
# ค้นหาเส้นทางจาก Alice ไปหา Roy โดยมีความยาวเส้นทางสูงสุดไม่เกิน 2
bfs_result = graph.bfs(fromExpr="id = 'Alice'", toExpr="id = 'Roy'", maxPathLength=2)  
bfs_result.show()

# หยุดการทำงานของ SparkSession เมื่อเสร็จสิ้น
spark.stop()
