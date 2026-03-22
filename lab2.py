# import pandas as pd
# import seaborn as sns 
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import stats

# #thiet lap kich thuoc mac dinh cho cac bieu do ngang 10 doc 6
# plt.rcParams['figure.figsize'] = (10,6)
# #chon phong cach nen bieu do nen trang co luoi
# sns.set_theme(style="whitegrid")

# #doc file csv
# df_housing = pd.read_csv('ITA105_Lab_2_lot.csv')
# # in bang thong ke (sl, avg, do lech chuan, min,max,cac moc 25-5--75%)
# print(df_housing.describe())
# # ve bieu do (boxplot) cho cot 'gia'
# #cac diem cham nam ngoai 'rau' cua hop chinh la ngoai le (outliers)
# sns.boxplot(x=df_housing['gia'])
# plt.title('Biểu đồ Boxplot của Giá Nhà')
# plt.show()

# #phuong phap IQR (Interquartile range)
# #gia nha tai moc 25%
# Q1 = df_housing['gia'].quantile(0.25)
# #gia nha tai moc 75%
# Q3 = df_housing['gia'].quantile(0.75)
# #khoang bien thien giua Q1 va Q3
# IQR = Q3 - Q1

# #tinh bien duoi va bien tren theo quy tac 1.5*IQR
# lower_bound = Q1 - 1.5 *IQR
# upper_bound = Q3 + 1.5 * IQR
# #------------------------------------------------
# # Lọc ra các dòng có giá thấp hơn biên dưới HOẶC cao hơn biên trên
# outliers_iqr = df_housing[(df_housing['gia'] < lower_bound) | (df_housing['gia'] > upper_bound)]
# print(f"Số lượng ngoại lệ theo IQR: {len(outliers_iqr)}")

# # 5. Phương pháp Z-score (Điểm tiêu chuẩn)
# # stats.zscore tính xem mỗi giá trị cách trung bình bao nhiêu lần độ lệch chuẩn
# # np.abs để lấy giá trị tuyệt đối (không phân biệt âm hay dương)
# df_housing['z_score_gia'] = np.abs(stats.zscore(df_housing['gia']))

# # Quy tắc 3-sigma: Nếu Z-score > 3 thì đó là điểm cực kỳ bất thường
# outliers_z = df_housing[df_housing['z_score_gia'] > 3]
# print(f"Số lượng ngoại lệ theo Z-score (>3): {len(outliers_z)}")

# # 6. Xử lý: Loại bỏ các dòng bị coi là ngoại lệ theo Z-score để làm sạch dữ liệu
# df_housing_clean = df_housing[df_housing['z_score_gia'] <= 3]


# #bai2
# # 1. Đọc dữ liệu, ép kiểu cột 'timestamp' sang dạng thời gian
# df_iot = pd.read_csv('ITA105_Lab_2_Iot.csv', parse_dates=['timestamp'])
# # Đặt cột thời gian làm chỉ mục (index) để dễ vẽ biểu đồ đường
# df_iot.set_index('timestamp', inplace=True)

# # 2. Vẽ biểu đồ đường để theo dõi sự thay đổi nhiệt độ theo thời gian
# df_iot['temperature'].plot()
# plt.title('Biến thiên nhiệt độ theo thời gian')
# plt.show()

# # 3. Phương pháp Rolling Mean (Trung bình trượt)
# window = 10 # Xét một nhóm 10 mẫu dữ liệu liên tiếp
# # Tính trung bình và độ lệch chuẩn của nhóm 10 mẫu này
# rolling_mean = df_iot['temperature'].rolling(window=window).mean()
# rolling_std = df_iot['temperature'].rolling(window=window).std()

# # Điểm bất thường là điểm lệch quá 3 lần độ lệch chuẩn so với trung bình của 10 mẫu trước đó
# upper_limit = rolling_mean + (3 * rolling_std)
# lower_limit = rolling_mean - (3 * rolling_std)

# # Lọc các điểm nằm ngoài ngưỡng trượt này
# outliers_iot = df_iot[(df_iot['temperature'] > upper_limit) | (df_iot['temperature'] < lower_limit)]

# # 4. Xử lý bằng Nội suy (Interpolation)
# df_iot['temp_cleaned'] = df_iot['temperature']
# # Bước A: Gán giá trị NaN (trống) vào những vị trí là ngoại lệ
# df_iot.loc[outliers_iot.index, 'temp_cleaned'] = np.nan
# # Bước B: Dùng hàm nội suy để tự động điền giá trị vào chỗ trống dựa trên các điểm xung quanh
# df_iot['temp_cleaned'] = df_iot['temp_cleaned'].interpolate(method='linear')

# #bai3
# df_ecom = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')

# # 1. Lọc theo logic thực tế: Giá không thể <= 0 và Rating không thể > 5
# logic_errors = df_ecom[(df_ecom['price'] <= 0) | (df_ecom['rating'] > 5)]
# print(f"Số lỗi logic: {len(logic_errors)}")

# # 2. Vẽ biểu đồ phân tán (Scatter plot) giữa Số lượng và Giá
# # Giúp ta thấy các đơn hàng mua số lượng cực lớn hoặc giá cực cao tách biệt ra
# sns.scatterplot(data=df_ecom, x='quantity', y='price')
# plt.title('Mối quan hệ giữa Số lượng và Giá')
# plt.show()

# # 3. Xử lý: Xóa bỏ các dòng lỗi logic đã tìm thấy ở trên
# df_ecom_clean = df_ecom.drop(logic_errors.index)

# #bai4
# # Vẽ biểu đồ phân tán diện tích và giá của tập Housing
# sns.scatterplot(data=df_housing, x='dien_tich', y='gia')
# # Vẽ thêm một đường đỏ đứt đoạn làm ngưỡng biên trên đã tính ở Bài 1
# plt.axhline(upper_bound, color='red', linestyle='--')
# plt.title('Phát hiện ngoại lệ đa biến: Diện tích vs Giá')
# plt.show()



# ==============================
# IMPORT THƯ VIỆN
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest


# ==============================
# BÀI 1: HOUSING DATA
# ==============================
print("===== BÀI 1: HOUSING =====")

# Load dữ liệu
df1 = pd.read_csv("housing.csv")

# Kiểm tra dữ liệu
print(df1.shape)
print(df1.isnull().sum())

# Thống kê mô tả
print(df1.describe())

# Boxplot
df1.select_dtypes(include=np.number).boxplot(figsize=(10,6))
plt.title("Boxplot Housing")
plt.show()

# Scatter
plt.scatter(df1['area'], df1['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# IQR
Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1

outlier_iqr = ((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR)))
print("Outlier IQR:\n", outlier_iqr.sum())

# Z-score
z = np.abs(stats.zscore(df1.select_dtypes(include=np.number)))
outlier_z = (z > 3)
print("Outlier Z-score:\n", outlier_z.sum())

# So sánh
print("Tổng IQR:", outlier_iqr.sum().sum())
print("Tổng Z-score:", outlier_z.sum().sum())

# Xử lý outlier (clipping)
df1_clean = df1.copy()
for col in df1.select_dtypes(include=np.number).columns:
    df1_clean[col] = np.clip(df1[col],
                            df1[col].quantile(0.05),
                            df1[col].quantile(0.95))

# Boxplot sau xử lý
df1_clean.boxplot(figsize=(10,6))
plt.title("After Cleaning")
plt.show()



# ==============================
# BÀI 2: IOT DATA
# ==============================
print("===== BÀI 2: IOT =====")

# Load dữ liệu
df2 = pd.read_csv("iot.csv", parse_dates=['timestamp'])
df2.set_index('timestamp', inplace=True)

print(df2.isnull().sum())

# Line plot
df2.plot(figsize=(12,6), title="IoT Data")
plt.show()

# Rolling mean + std
rolling_mean = df2.rolling(window=10).mean()
rolling_std = df2.rolling(window=10).std()

outlier_roll = ((df2 > rolling_mean + 3*rolling_std) |
                (df2 < rolling_mean - 3*rolling_std))

print("Outlier Rolling:\n", outlier_roll.sum())

# Z-score
z2 = np.abs(stats.zscore(df2))
print("Outlier Z-score:\n", (z2 > 3).sum())

# Scatter
sns.scatterplot(x=df2['temperature'], y=df2['pressure'])
plt.title("Temp vs Pressure")
plt.show()

# Xử lý (interpolation)
df2_clean = df2.interpolate()

df2_clean.plot(title="After Cleaning")
plt.show()



# ==============================
# BÀI 3: E-COMMERCE
# ==============================
print("===== BÀI 3: E-COMMERCE =====")

# Load dữ liệu
df3 = pd.read_csv("ecommerce.csv")

print(df3.describe())
print(df3.isnull().sum())

# Boxplot
sns.boxplot(data=df3[['price','quantity','rating']])
plt.title("Boxplot E-commerce")
plt.show()

# IQR
Q1 = df3[['price','quantity','rating']].quantile(0.25)
Q3 = df3[['price','quantity','rating']].quantile(0.75)
IQR = Q3 - Q1

outlier3 = ((df3[['price','quantity','rating']] < (Q1 - 1.5 * IQR)) |
            (df3[['price','quantity','rating']] > (Q3 + 1.5 * IQR)))

print("Outlier:\n", outlier3.sum())

# Scatter
plt.scatter(df3['price'], df3['quantity'])
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.title("Price vs Quantity")
plt.show()

# Xử lý dữ liệu
df3 = df3[df3['price'] > 0]
df3 = df3[df3['rating'] <= 5]

df3['price'] = np.clip(df3['price'],
                       df3['price'].quantile(0.05),
                       df3['price'].quantile(0.95))

# Boxplot sau xử lý
sns.boxplot(data=df3[['price','quantity','rating']])
plt.title("After Cleaning")
plt.show()



# ==============================
# BÀI 4: MULTIVARIATE OUTLIER
# ==============================
print("===== BÀI 4: MULTIVARIATE =====")

df4 = pd.read_csv("data.csv")

model = IsolationForest(contamination=0.05, random_state=42)
df4['outlier'] = model.fit_predict(df4.select_dtypes(include=np.number))

print(df4['outlier'].value_counts())

# Scatter
sns.scatterplot(x=df4.iloc[:,0], y=df4.iloc[:,1], hue=df4['outlier'])
plt.title("Isolation Forest Result")
plt.show()