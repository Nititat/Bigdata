# นำเข้าไลบรารีที่จำเป็นจาก PySpark สำหรับการทำงานกับ DataFrame, การจัดประเภท (classification), และการประเมินผล (evaluation)
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# สร้าง SparkSession ซึ่งเป็นจุดเริ่มต้นของการทำงานกับ PySpark
spark = SparkSession.builder.appName("DecisionTreeExample").getOrCreate()

# โหลดข้อมูลจากไฟล์ CSV ลงใน DataFrame
# ตัวแปร `data` เก็บข้อมูลที่ถูกโหลดจากไฟล์ CSV (fb_live_thailand.csv)
# header=True หมายถึงไฟล์ CSV มีบรรทัดแรกเป็นหัวข้อคอลัมน์
# inferSchema=True หมายถึงให้ Spark เดาชนิดของข้อมูลในแต่ละคอลัมน์โดยอัตโนมัติ
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# ตรวจสอบ schema ของข้อมูลว่าแต่ละคอลัมน์ถูกกำหนดชนิดข้อมูลถูกต้องหรือไม่
data.printSchema()

# แปลงคอลัมน์ที่เป็นข้อมูลประเภทตัวอักษร (categorical) ให้เป็นตัวเลข (index) ด้วย StringIndexer
# status_type_indexer: ตัวแปรที่เก็บกระบวนการแปลงคอลัมน์ `status_type` ให้เป็นตัวเลข (index) ในคอลัมน์ใหม่ชื่อ `status_type_ind`
status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind", handleInvalid="keep")

# status_published_indexer: ตัวแปรที่เก็บกระบวนการแปลงคอลัมน์ `status_published` ให้เป็นตัวเลขในคอลัมน์ใหม่ชื่อ `status_published_ind`
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind", handleInvalid="keep")

# แปลงคอลัมน์ที่เป็นตัวเลข (index) ให้เป็น one-hot encoded vectors ด้วย OneHotEncoder
# status_type_encoder: แปลงคอลัมน์ `status_type_ind` ให้เป็น one-hot encoded vector ในคอลัมน์ใหม่ `status_type_vec`
status_type_encoder = OneHotEncoder(inputCols=["status_type_ind"], outputCols=["status_type_vec"])

# status_published_encoder: แปลงคอลัมน์ `status_published_ind` ให้เป็น one-hot encoded vector ในคอลัมน์ใหม่ `status_published_vec`
status_published_encoder = OneHotEncoder(inputCols=["status_published_ind"], outputCols=["status_published_vec"])

# รวมฟีเจอร์ (คอลัมน์ที่ใช้ในการทำนาย) เข้าด้วยกันเป็นฟีเจอร์เดียวในคอลัมน์ `features` ด้วย VectorAssembler
# assembler: ตัวแปรที่เก็บขั้นตอนการรวมคอลัมน์ฟีเจอร์ที่ผ่านการแปลงเป็นเวกเตอร์ (เช่น `status_type_vec` และ `status_published_vec`)
assembler = VectorAssembler(
    inputCols=["status_type_vec", "status_published_vec"],  # ใช้คอลัมน์ที่ถูกแปลงเป็นเวกเตอร์
    outputCol="features"  # ฟีเจอร์ที่รวมกันแล้วเก็บในคอลัมน์ชื่อ `features`
)

# สร้างโมเดล Decision Tree สำหรับการจัดประเภท
# dt: ตัวแปรที่เก็บโมเดล Decision Tree ซึ่งใช้ในการจำแนกประเภท (classification)
# featuresCol คือคอลัมน์ที่เป็นฟีเจอร์ (ข้อมูลที่ใช้ในการทำนาย)
# labelCol คือคอลัมน์ที่เป็นเป้าหมาย (ค่าที่ต้องการทำนาย)
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="status_type_ind"  # คอลัมน์ที่เป็นเป้าหมาย (label)
)

# สร้าง Pipeline เพื่อรวมขั้นตอนการทำ indexing, encoding, การรวมฟีเจอร์ และการสร้างโมเดลเข้าด้วยกัน
# pipeline: ตัวแปรที่เก็บ Pipeline ซึ่งรวมขั้นตอนการแปลงข้อมูลและการสร้างโมเดล
pipeline = Pipeline(stages=[status_type_indexer, status_published_indexer, status_type_encoder, status_published_encoder, assembler, dt])

# แบ่งข้อมูลออกเป็นชุดฝึก (train_data) และชุดทดสอบ (test_data)
# train_data และ test_data เก็บข้อมูลที่แบ่งออกเป็น 70% สำหรับฝึกโมเดล และ 30% สำหรับทดสอบโมเดล
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)  # seed=42 ใช้เพื่อให้ผลลัพธ์สามารถทำซ้ำได้

# ฝึกโมเดล Decision Tree โดยใช้ข้อมูลชุดฝึก (train_data)
# model: ตัวแปรที่เก็บโมเดล Decision Tree ที่ผ่านการฝึกแล้ว
model = pipeline.fit(train_data)

# ใช้โมเดลที่ฝึกแล้วทำนายผลลัพธ์จากข้อมูลชุดทดสอบ (test_data)
# predictions: ตัวแปรที่เก็บผลลัพธ์การทำนายจากข้อมูล test_data
predictions = model.transform(test_data)

# แสดงผลลัพธ์การทำนาย 5 แถวแรก โดยแสดงคอลัมน์ `status_type_ind` (ค่าจริง), `prediction` (ค่าทำนาย), และ `probability` (ความน่าจะเป็น)
predictions.select("status_type_ind", "prediction", "probability").show(5)

# สร้างตัวประเมินผลลัพธ์ของโมเดล (evaluator) สำหรับการจัดประเภทแบบหลายคลาส (multiclass classification)
# evaluator: ตัวแปรที่เก็บ MulticlassClassificationEvaluator สำหรับใช้ประเมินความถูกต้องของโมเดล
evaluator = MulticlassClassificationEvaluator(
    labelCol="status_type_ind",  # คอลัมน์ที่เป็น label (ค่าจริง)
    predictionCol="prediction"   # คอลัมน์ที่เป็นค่าทำนาย
)

# คำนวณค่า accuracy (ความถูกต้อง), precision (ความแม่นยำ), recall (ความครอบคลุม) และ F1 score ของโมเดล
# accuracy เก็บค่าความถูกต้องของโมเดล โดยเทียบผลลัพธ์ที่ทำนายกับค่าจริง
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
# precision เก็บค่าความแม่นยำของโมเดล โดยดูจากจำนวนที่ทำนายถูกจากจำนวนที่ทำนายทั้งหมด
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
# recall เก็บค่าความครอบคลุม โดยดูจากจำนวนที่ทำนายถูกจากจำนวนที่เป็นค่าจริง
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
# F1 score เป็นค่าที่ผสมระหว่าง precision และ recall เพื่อวัดประสิทธิภาพของโมเดล
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

# แสดงค่าที่คำนวณได้จากตัวประเมินผลลัพธ์ (accuracy, precision, recall, และ F1 score)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Measure: {f1}")

# คำนวณและแสดงค่า Test Error (ค่า Error จากข้อมูลทดสอบ)
# test_error: ตัวแปรที่เก็บค่าความผิดพลาดในการทำนาย โดยคำนวณจาก 1 ลบด้วยค่า accuracy
test_error = 1.0 - accuracy
print(f"Test Error: {test_error}")

# หยุดการทำงานของ SparkSession หลังจากเสร็จสิ้นการประมวลผล
spark.stop()
