# นำเข้าไลบรารีที่จำเป็นจาก PySpark สำหรับการทำงานกับ DataFrame, Logistic Regression และการประเมินผล
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# สร้าง SparkSession ซึ่งเป็นจุดเริ่มต้นของการทำงานกับ PySpark
spark = SparkSession.builder.appName("LogisticRegressionExample").getOrCreate()

# โหลดข้อมูลจากไฟล์ CSV ลงใน DataFrame
# data เป็นตัวแปรที่เก็บข้อมูลที่ถูกโหลดจากไฟล์ fb_live_thailand.csv
# header=True หมายถึงไฟล์ CSV มีบรรทัดแรกเป็นหัวข้อคอลัมน์
# inferSchema=True หมายถึงให้ Spark เดาว่าข้อมูลแต่ละคอลัมน์ควรเป็นชนิดข้อมูลแบบไหน
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# ตรวจสอบ schema ของข้อมูลว่าคอลัมน์ถูกกำหนดชนิดข้อมูลถูกต้องหรือไม่
# printSchema() จะแสดงประเภทของข้อมูลแต่ละคอลัมน์ใน DataFrame
data.printSchema()

# แปลงคอลัมน์ที่เป็นข้อมูลประเภทตัวอักษร (categorical) ให้เป็นตัวเลข (index)
# StringIndexer จะแปลงค่าที่เป็นตัวอักษรเป็นตัวเลข เช่น "video" อาจถูกแปลงเป็น 0, "photo" อาจถูกแปลงเป็น 1
# status_type_indexer เป็นตัวแปรที่เก็บกระบวนการแปลงคอลัมน์ "status_type" ให้เป็นตัวเลข และเก็บผลลัพธ์ในคอลัมน์ใหม่ชื่อ "status_type_ind"
status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind", handleInvalid="keep")

# คอลัมน์ "status_published" ที่เป็นเวลาถูกแปลงเป็นตัวเลขในคอลัมน์ "status_published_ind"
# status_published_indexer เป็นตัวแปรที่เก็บกระบวนการแปลงคอลัมน์ "status_published"
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind", handleInvalid="keep")

# รวมคอลัมน์ที่ต้องการใช้เป็นฟีเจอร์ (input) ในการฝึกโมเดล
# assembler เป็นตัวแปรที่เก็บกระบวนการรวมคอลัมน์ที่ผ่านการแปลง เช่น 'status_type_ind' และ 'status_published_ind' เข้าเป็นฟีเจอร์เดียวในคอลัมน์ใหม่ชื่อว่า "features"
assembler = VectorAssembler(
    inputCols=["status_type_ind", "status_published_ind"],  # คอลัมน์ที่ต้องการรวมเป็นฟีเจอร์
    outputCol="features"  # คอลัมน์ใหม่ที่จะเก็บฟีเจอร์ที่รวมกันแล้ว
)

# สร้างโมเดล Logistic Regression
# lr เป็นตัวแปรที่เก็บโมเดล Logistic Regression ซึ่งเป็นโมเดลสำหรับการจำแนกประเภท (classification)
# Logistic Regression เป็นโมเดลที่ใช้ในการทำนายข้อมูลแบบจำแนกประเภท เช่น ทำนายว่าข้อมูลเป็นหมวดหมู่ไหน
lr = LogisticRegression(
    featuresCol="features",  # คอลัมน์ที่เป็นฟีเจอร์ (input) สำหรับโมเดล
    labelCol="status_type_ind",  # คอลัมน์ที่เป็นเป้าหมาย (label) ที่ต้องการทำนาย
    maxIter=10,           # จำนวนรอบสูงสุดในการทำซ้ำเพื่อปรับปรุงโมเดล
    regParam=0.01,        # ค่าพารามิเตอร์สำหรับ regularization เพื่อป้องกัน overfitting
    elasticNetParam=0.8   # ค่าผสมระหว่าง L1 และ L2 regularization (ElasticNet)
)

# สร้าง Pipeline ซึ่งใช้รวมขั้นตอนการทำงานหลายขั้นตอนเข้าด้วยกัน
# pipeline เป็นตัวแปรที่เก็บ Pipeline ซึ่งรวมการทำ indexing, การรวมฟีเจอร์ และการสร้างโมเดล Logistic Regression เข้าเป็นขั้นตอนเดียว
pipeline = Pipeline(stages=[status_type_indexer, status_published_indexer, assembler, lr])

# แบ่งข้อมูลออกเป็นสองชุดคือชุดฝึก (train_data) และชุดทดสอบ (test_data)
# train_data และ test_data เก็บข้อมูลที่แบ่งออกเป็น 70% สำหรับการฝึกโมเดล และ 30% สำหรับการทดสอบ
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)  # seed=42 ใช้เพื่อให้ผลลัพธ์สามารถทำซ้ำได้

# ฝึกโมเดล Logistic Regression โดยใช้ข้อมูลชุดฝึก
# model เป็นตัวแปรที่เก็บโมเดล Logistic Regression ที่ผ่านการฝึกจากข้อมูล train_data แล้ว
model = pipeline.fit(train_data)

# ใช้โมเดลที่ฝึกเสร็จแล้วในการทำนายผลลัพธ์จากข้อมูลชุดทดสอบ
# predictions เป็นตัวแปรที่เก็บผลลัพธ์ที่ได้จากการทำนายโดยใช้ข้อมูล test_data
predictions = model.transform(test_data)

# แสดงผลลัพธ์การทำนาย 5 แถวแรก โดยแสดงคอลัมน์ "status_type_ind" (ค่าจริง), "prediction" (ค่าทำนาย), และ "probability" (ความน่าจะเป็น)
predictions.select("status_type_ind", "prediction", "probability").show(5)

# สร้างตัวประเมินผลลัพธ์ของโมเดล (evaluator) สำหรับการจัดประเภทแบบหลายคลาส
# evaluator เป็นตัวแปรที่เก็บ MulticlassClassificationEvaluator สำหรับใช้ประเมินความถูกต้องของโมเดล
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

# หยุดการทำงานของ SparkSession หลังจากเสร็จสิ้นการประมวลผล
spark.stop()
