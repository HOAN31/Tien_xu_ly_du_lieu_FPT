import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




df = pd.read_csv('bat_dong_san_raw.csv')

print(df.head())
print(df.info())

print(df.describe())
print(df.median(numeric_only=True))

print("Missing:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

sns.histplot(df['gia_nha'], kde=True)
plt.title("Phan phoi gia nha")
plt.show()

sns.boxplot(x=df['gia_nha'])
plt.title("Outliers gia nha")
plt.show()

#2
#xu ly thieu gia tri
df['gia_nha'].fillna(df['gia_nha'].median(), inplace=True)
df['tinh_trang'].fillna(df['tinh_trang'].mode()[0], inplace=True)

#xoa du lieu sai
df = df[df['gia_nha'] > 0]
df = df[df['dien_tich'] > 0]
df = df[df['so_phong'] > 0]

#xu ly trung lap
df = df.drop_duplicates()

#chuan hoa text
df['vi_tri'] = df['vi_tri'].str.strip().str.lower()

#3
#phat hien outlier (IQR)
Q1 = df['gia_nha'].quantile(0.25)
Q3 = df['gia_nha'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['gia_nha'] >=Q1 - 1.5*IQR) &
        (df['gia_nha'] <=Q3 + 1.5*IQR)]

# giam lech (skew)
df['gia_nha_log'] = np.log1p(df['gia_nha'])

#4
#chuan hoa so
scaner = StandardScaler()
df[['gia-nha','dien_tich','so_phong']] = scaner.fit_transform(
    df[['gia_nha','dien_tich','so_phong']]
    )
#encode du lieu chu
df = pd.get_dummies(df, columns=['vi_tri'],drop_first=True)

#5 phat hien trung lap bang text (don gian)

tfidf = TfidfVectorizer()
text_matrix = tfidf.fit_transform(df['tinh_trang'].astype(str))

similarity = cosine_similarity(text_matrix)
print(similarity)