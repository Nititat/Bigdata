import pandas as pd
from pmdarima.arima import auto_arima, ADFTest
import matplotlib.pyplot as plt

# 1. นำเข้าข้อมูลจากไฟล์ year_sales.csv
df = pd.read_csv('year_sales.csv')

# 2. ใช้ฟังก์ชัน to_datetime() เพื่อแปลงคอลัมน์ Year เป็น datetime
df['Year'] = pd.to_datetime(df['Year'])

# 3. ตั้งค่า Year เป็นดัชนีโดยใช้ฟังก์ชัน set_index()
df.set_index('Year', inplace=True)

# 4. พล็อตข้อมูลดั้งเดิม
df.plot()
plt.title('Yearly Sales')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()

# 5. ตรวจสอบว่าจำเป็นต้องทำ Differencing หรือไม่ด้วย ADFTest
adf_test = ADFTest(alpha=0.05)
should_diff = adf_test.should_diff(df)

# 6. แบ่งข้อมูลเพื่อใช้ในการฝึกฝน (80%) และทดสอบ (20%)
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# 7. สร้างโมเดล ARIMA โดยใช้ auto_arima
arima_model = auto_arima(train, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, 
                         start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, 
                         m=12, seasonal=True, error_action='warn', trace=True, 
                         suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)

# 8. แสดงสรุปของโมเดล ARIMA
print(arima_model.summary())

# 9. สร้างการคาดการณ์จากโมเดล ARIMA
n_periods = len(test)
prediction = pd.DataFrame(arima_model.predict(n_periods=n_periods), index=test.index)
prediction.columns = ['Predicted']

# 10. พล็อตข้อมูล train, test, และผลการคาดการณ์
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(prediction, label='Predicted', color='red')
plt.title('Time Series Analysis - Train, Test, and Predicted')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.show()