from pyspark.sql import SparkSession  # ใช้สำหรับสร้าง Spark session เพื่อทำงานกับ Spark SQL
from pyspark.sql.types import IntegerType  # ใช้สำหรับแปลงข้อมูลให้เป็นชนิด Integer
from pyspark.sql.functions import trim, col  # ใช้ในการตัดช่องว่างและเลือกคอลัมน์
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover  # เครื่องมือสำหรับการสร้างฟีเจอร์
from pyspark.ml import Pipeline  # ใช้สำหรับสร้าง Pipeline สำหรับขั้นตอนการประมวลผล
from pyspark.ml.classification import LogisticRegression  # โมเดล Logistic Regression สำหรับการจำแนกประเภท
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  # ตัวประเมินผลสำหรับคำนวณความแม่นยำของโมเดล
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder  # สำหรับปรับแต่งพารามิเตอร์ด้วย Cross-Validation

# สร้าง Spark session
spark = SparkSession.builder \
    .appName("Text Classification") \
    .config("spark.executor.memory", "8g")\
    .config("spark.driver.memory", "8g")\
    .config("spark.driver.extraJavaOptions", "-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.NativeRefBLAS")\
    .getOrCreate()  

# อ่านข้อมูลจากไฟล์ CSV
file_path = "reviews_rated.csv"  # กำหนดเส้นทางของไฟล์ CSV
data = spark.read.csv(file_path, header=True, inferSchema=True)  # อ่านไฟล์ CSV พร้อมทั้งกำหนดว่ามีส่วน header และให้เดาชนิดของข้อมูล

# เลือกและประมวลผลคอลัมน์ Review Text และ Rating
data = data.select(
    trim(col("Review Text")).alias("ReviewText"),  # ตัดช่องว่างจากคอลัมน์ "Review Text" และเปลี่ยนชื่อเป็น "ReviewText"
    col("Rating").cast(IntegerType()).alias("Rating")  # แปลงคอลัมน์ "Rating" เป็นชนิด Integer และเปลี่ยนชื่อเป็น "Rating"
)

# กรองออกข้อมูลที่ ReviewText หรือ Rating เป็น null หรือว่างเปล่า
data = data.filter(
    (col("ReviewText").isNotNull()) & (col("ReviewText") != "") & 
    (col("Rating").isNotNull()) & (col("Rating") != float('nan'))
)

# แสดงข้อมูล
data.show(truncate=False)  # แสดงข้อมูลที่ประมวลผลแล้วโดยไม่ตัดทอนเนื้อหา

# Tokenizer
tokenizer = Tokenizer(inputCol="ReviewText", outputCol="ReviewTextWords")  # แยกข้อความรีวิวเป็นคำ ๆ

# StopWordsRemover
stop_word_remover = StopWordsRemover(
    inputCol=tokenizer.getOutputCol(),  # ใช้ข้อความที่ผ่านการแยกคำเป็นข้อมูลนำเข้า
    outputCol="MeaningfulWords"  # แสดงผลเป็นคำที่มีความหมาย (ลบคำที่เป็น stopwords ออก)
)

# HashingTF
hashing_tf = HashingTF(
    inputCol=stop_word_remover.getOutputCol(),  # ใช้คำที่มีความหมายเป็นข้อมูลนำเข้า
    outputCol="features"  # แสดงผลเป็นเวกเตอร์ฟีเจอร์
)

# Pipeline
pipeline = Pipeline(stages=[tokenizer, stop_word_remover, hashing_tf])  # กำหนดลำดับขั้นตอนการประมวลผล

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)  # แบ่งข้อมูลเป็น 80% สำหรับฝึก และ 20% สำหรับทดสอบ

# แสดงข้อมูลชุดฝึก
train_data.show(truncate=False)  # แสดงข้อมูลชุดฝึกโดยไม่ตัดเนื้อหา

# ฝึก Pipeline ด้วยข้อมูลชุดฝึก
pipeline_model = pipeline.fit(train_data)  # ฝึก Pipeline ด้วยข้อมูลชุดฝึก

# แปลงข้อมูลชุดฝึกและชุดทดสอบ
train_df = pipeline_model.transform(train_data)  # แปลงข้อมูลชุดฝึกด้วย Pipeline ที่ผ่านการฝึกแล้ว
test_df = pipeline_model.transform(test_data)  # แปลงข้อมูลชุดทดสอบด้วย Pipeline ที่ผ่านการฝึกแล้ว

# แสดงข้อมูลชุดฝึกที่แปลงแล้ว
train_df.show(truncate=False)  # แสดงข้อมูลชุดฝึกที่ผ่านการแปลงแล้ว

# Logistic Regression
lr = LogisticRegression(labelCol="Rating", featuresCol="features")  # สร้างโมเดล Logistic Regression

# สร้าง Grid ของพารามิเตอร์
paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10, 20]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

# สร้าง CrossValidator
crossval = CrossValidator(estimator=lr,  # โมเดลที่ต้องการปรับแต่ง
                          estimatorParamMaps=paramGrid,  # พารามิเตอร์ที่ต้องการทดสอบ
                          evaluator=MulticlassClassificationEvaluator(labelCol="Rating", predictionCol="prediction", metricName="accuracy"),  # ตัวประเมินผล
                          numFolds=5)  # จำนวน folds สำหรับ cross-validation

# ฝึกโมเดลด้วย CrossValidator
cv_model = crossval.fit(train_df)  # ฝึกโมเดลด้วยข้อมูลชุดฝึก

# ทำนายข้อมูลชุดทดสอบ
predictions = cv_model.transform(test_df)  # ทำการทำนายชุดทดสอบ

# แสดง MeaningfulWords, Rating (label) และ Prediction
predictions.select("MeaningfulWords", "Rating", "prediction").show(truncate=False)  # แสดงคำที่มีความหมาย, ค่าจริงของเรตติ้ง, และค่าที่ทำนายได้

# สร้างตัวประเมินผล MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="Rating",  # ระบุคอลัมน์ของ label
    predictionCol="prediction",  # ระบุคอลัมน์ของการทำนาย
    metricName="accuracy"  # กำหนดค่าที่ต้องการวัดเป็นความแม่นยำ (accuracy)
)
print('')
print('='*80)
print('')
# ประเมินความแม่นยำของโมเดล
accuracy = evaluator.evaluate(predictions)  # คำนวณความแม่นยำของโมเดลบนข้อมูลชุดทดสอบ
print(f"Model Accuracy: {accuracy:.4f}")  # แสดงค่าความแม่นยำ

# แสดงความแม่นยำของโมเดลที่ใช้ Cross-Validation
cv_accuracy = evaluator.evaluate(predictions)  # คำนวณความแม่นยำ
print(f"Cross-Validated Model Accuracy: {cv_accuracy:.4f}")  # แสดงค่าความแม่นยำ
print('')
print('='*80)
print('')