import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#  BÀI 1: HOUSING
print("===== BÀI 1 =====")

df = pd.read_csv("housing.csv")

# 1. shape + missing
print("Shape:", df.shape)
print("Missing:\n", df.isnull().sum())

# 2. thống kê
print(df.describe())

# lấy các cột số
data = df.select_dtypes(include=np.number)

# 3. boxplot
data.boxplot()
plt.title("Boxplot Housing")
plt.show()

# 4. scatter
plt.scatter(data.iloc[:,0], data.iloc[:,1])
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title("Scatter Housing")
plt.show()

# 5. IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outlier_iqr = ((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR)))
print("Outlier IQR:\n", outlier_iqr.sum())

# 6. Z-score
z = np.abs(stats.zscore(data))
outlier_z = (z > 3)
print("Outlier Z:\n", outlier_z.sum())

# 7. so sánh
print("So sánh IQR vs Z:")
print("IQR:", outlier_iqr.sum().sum())
print("Z-score:", outlier_z.sum().sum())

# 8. nhận xét
print("Nhận xét: Ngoại lệ có thể do giá nhà cao bất thường hoặc lỗi nhập liệu")

# 9. xử lý (clip theo IQR)
data_clean = data.copy()
for col in data.columns:
    lower = Q1[col] - 1.5*IQR[col]
    upper = Q3[col] + 1.5*IQR[col]
    data_clean[col] = data[col].clip(lower, upper)

# 10. vẽ lại
data_clean.boxplot()
plt.title("After Cleaning Housing")
plt.show()


#  BÀI 2: IoT
print("===== BÀI 2 =====")

df = pd.read_csv("iot.csv")

# 1. timestamp + missing
df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
df.set_index(df.columns[0], inplace=True)

print("Missing:\n", df.isnull().sum())

data = df.select_dtypes(include=np.number)

# 2. line plot
data.plot()
plt.title("IoT Line Plot")
plt.show()

# 3. rolling
mean = data.rolling(10).mean()
std = data.rolling(10).std()

outlier_roll = (data > mean + 3*std) | (data < mean - 3*std)
print("Outlier Rolling:\n", outlier_roll.sum())

# 4. Z-score
z = np.abs(stats.zscore(data))
outlier_z = (z > 3)
print("Outlier Z:\n", outlier_z.sum())

# 5. boxplot + scatter
data.boxplot()
plt.title("Boxplot IoT")
plt.show()

plt.scatter(data.iloc[:,0], data.iloc[:,1])
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title("Scatter IoT")
plt.show()

# 6. so sánh
print("So sánh Rolling vs Z:")
print("Rolling:", outlier_roll.sum().sum())
print("Z-score:", outlier_z.sum().sum())

# 7. xử lý (interpolate)
data_clean = data.interpolate()

data_clean.plot()
plt.title("After Cleaning IoT")
plt.show()


#  BÀI 3: E-COMMERCE
print("===== BÀI 3 =====")

df = pd.read_csv("ecommerce.csv")

# 1.
print("Missing:\n", df.isnull().sum())
print(df.describe())

data = df.select_dtypes(include=np.number)

# 2.
data.boxplot()
plt.title("Boxplot Ecommerce")
plt.show()

# 3. IQR + Z
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outlier_iqr = ((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR)))
print("Outlier IQR:\n", outlier_iqr.sum())

z = np.abs(stats.zscore(data))
outlier_z = (z > 3)
print("Outlier Z:\n", outlier_z.sum())

# 4. scatter
plt.scatter(data.iloc[:,0], data.iloc[:,1])
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title("Scatter Ecommerce")
plt.show()

# 5. nhận xét
print("Nguyên nhân: giá 0, rating >5 hoặc số lượng bất thường")

# 6. xử lý
data_clean = data.copy()

# bỏ giá trị âm hoặc 0
data_clean = data_clean[data_clean.iloc[:,0] > 0]

# clip từng cột (FIX LỖI)
for col in data.columns:
    max_val = data[col].quantile(0.99)
    data_clean[col] = data_clean[col].clip(upper=max_val)

# 7. vẽ lại
data_clean.boxplot()
plt.title("After Cleaning Ecommerce")
plt.show()

plt.scatter(data_clean.iloc[:,0], data_clean.iloc[:,1])
plt.xlabel(data_clean.columns[0])
plt.ylabel(data_clean.columns[1])
plt.title("Scatter After Cleaning")
plt.show()


#  BÀI 4: MULTIVARIATE
print("===== BÀI 4 =====")

def multi(file):
    df = pd.read_csv(file)
    data = df.select_dtypes(include=np.number)

    z = np.abs(stats.zscore(data))
    outliers = (z > 3).any(axis=1)

    plt.scatter(data.iloc[:,0], data.iloc[:,1], c=outliers)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title(file)
    plt.show()

multi("housing.csv")
multi("iot.csv")
multi("ecommerce.csv")

print("Nhận xét: Multivariate phát hiện ngoại lệ tốt hơn vì xét nhiều biến cùng lúc")
