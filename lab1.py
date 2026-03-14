import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ITA105_Lab_1.csv')
print("---Kích thước của dữu liệu---")
print("Số dòng: ",df.shape[0])
print("Số cột:",df.shape[1])

print("---Thông kê mô tả ---")

print(df.describe())

print("---Kiểm tra giá trị thiếu ---")
print(df.isnull().sum())

#Phát hiện giá trị thiếu:
missing_val = df.isnull().sum()
# Điền cột Price bằng trung bình (mean)
df['Price'] = df['Price'].fillna(df['Price'].mean())

# Điền cột StockQuantity bằng trung vị (median)
df['StockQuantity'] = df['StockQuantity'].fillna(df['StockQuantity'].median())

# Điền cột Category bằng yếu vị (mode) - lấy giá trị đầu tiên [0]
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])
df_dropped = df.dropna()
print(f"Số dòng sau khi điền: {len(df)}")
print(f"Số dòng nếu dùng dropna(): {len(df_dropped)}")

#bài3
#Xử lý giá trị bất hợp lý trong Price và StockQuantity (loại bỏ giá trị âm):
df = df[df['Price'] >= 0]
df = df[df['StockQuantity'] >= 0]
#Lọc các giá trị không hợp lệ trong cột Rating (giả sử Rating từ 1 đến 5):
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)]

#bài4
#Áp dụng Moving Average (Trung bình trượt) cho cột Price:
# window=3 nghĩa là tính trung bình của 3 dòng liên tiếp
df['Price_Smooth'] = df['Price'].rolling(window=3).mean()
plt.figure(figsize=(10,6))
plt.plot(df['Price'], label='Giá gốc (Gốc)', alpha=0.5)
plt.plot(df['Price_Smooth'], label='Giá đã làm mượt (MA)', color='red')
plt.legend()
plt.title("So sánh giá trước và sau khi làm mượt")
plt.show()


#bai 5
#Chuyển Category thành chữ thường:
df['Category'] = df['Category'].str.lower()
#Loại bỏ ký tự thừa (khoảng trắng) trong Description:
df['Description'] = df['Description'].str.strip()
#Loại bỏ ký tự thừa (khoảng trắng) trong Description:
df['Price_VND'] = df['Price'] * 25000