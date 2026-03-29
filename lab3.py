
import pandas as pd
#thư viện để xử lý dữ liệu dạng bảng, hỗ trọ đọc file csv, tính toán thông kê và trực quan hóa dữu liệu
import numpy as np
#thư viện hỗ trợ tính toán số học, đại số tuyến tính, và các phép toán trên mảng đa chiều
import matplotlib.pyplot as plt
#thư viện để vẽ biểu đồ, đồ thị
import seaborn as sns 
#thư viện dụa trên matplotlib, giao diện cao cấp hơn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# sklearn.preprocessing cung cấp các công cụ để chuẩn hóa dữ liệu, MinMaxScaLer đưa dữ liệu về khoản [0,1], StandardScaler chuẩn hóa dữu liệu về mean=0, std=1

sns.set(style="whitegrid")
#Thiết lập giao diện mặc định cho biểu đồ

# BÀI 1: THÔNG SỐ VẬN ĐỘNG VIÊN (Sports Data)
print("--- Đang xử lý Bài 1 ---")
df_sports = pd.read_csv('ITA105_Lab_3_Sports.csv')
# đọc file csv tên ...
print(df_sports.isnull().sum())
# kiếm tra số lượng giá trị bị thiếu trong các cột
print(df_sports.describe())
# thống kê cơ bản về dữ liệu bao gồm: sl, mean, std, min max, các phần trăm phân vị 25-50-75%


scaler_mm = MinMaxScaler()
# đưa dữ liệu về khaonr [0,1] bằng công thức: (x-min)/(max-min)
scaler_zs = StandardScaler()
#chuẩn hóa dữu liệu về mean=0, std=1 bằng công thức: (x-mean)/std

cols_sports = ['chieu_cao_cm', 'can_nang_kg', 'toc_do_100m_s', 'so_ban_thang', 'so_phut_thi_dau']
# chọn các cột cần chuẩn hóa

df_sports_mm = pd.DataFrame(scaler_mm.fit_transform(df_sports[cols_sports]), columns=cols_sports)
# áp dụng min-max scalder cho các cột đã chọn, tạo DataFrame mới
df_sports_zs = pd.DataFrame(scaler_zs.fit_transform(df_sports[cols_sports]), columns=cols_sports)
# áp dụng  standard scaler cho các cột đã chọn, tạo DataFrame mới



# Vẽ biểu đồ so sánh phân phối trước và sau chuẩn hóa (Biến tốc độ 100m) [cite: 11]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df_sports['toc_do_100m_s'], kde=True, ax=axes[0], color='skyblue').set_title("Gốc (Original)")
sns.histplot(df_sports_mm['toc_do_100m_s'], kde=True, ax=axes[1], color='salmon').set_title("Min-Max Scaling")
sns.histplot(df_sports_zs['toc_do_100m_s'], kde=True, ax=axes[2], color='green').set_title("Z-Score Normalization")
plt.tight_layout()
plt.show()

# =================================================================
# BÀI 2: CHỈ SỐ BỆNH NHÂN (Health Data) [cite: 12]
# =================================================================
print("\n--- Đang xử lý Bài 2 ---")
df_health = pd.read_csv('ITA105_Lab_3_Health.csv')

# Trực quan hóa để phát hiện ngoại lệ (Outliers) [cite: 13, 14]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_health)
plt.title("Phát hiện ngoại lệ trong chỉ số sức khỏe")
plt.show()

# NHẬN XÉT BÀI 2[cite: 17]:
# - Biến 'huyet_ap_mmHg' bị ảnh hưởng nhiều nhất bởi ngoại lệ (có giá trị lên tới 250).
# - Phương pháp Z-Score phù hợp hơn vì nó không ép toàn bộ dữ liệu vào khoảng hẹp [0, 1] 
#   khi có giá trị cực đoan, giúp giữ lại đặc điểm phân phối tốt hơn.

# =================================================================
# BÀI 3: CHỈ SỐ CÔNG TY (Finance Data) [cite: 18]
# =================================================================
print("\n--- Đang xử lý Bài 3 ---")
df_finance = pd.read_csv('ITA105_Lab_3_Finance.csv')

# Vẽ Scatterplot so sánh Doanh thu và Lợi nhuận trước/sau chuẩn hóa [cite: 21, 22]
# Chuẩn hóa nhanh cho Bài 3 [cite: 20]
df_fin_mm = pd.DataFrame(scaler_mm.fit_transform(df_finance[['doanh_thu_musd', 'loi_nhuan_musd']]), 
                         columns=['doanh_thu', 'loi_nhuan'])
df_fin_zs = pd.DataFrame(scaler_zs.fit_transform(df_finance[['doanh_thu_musd', 'loi_nhuan_musd']]), 
                         columns=['doanh_thu', 'loi_nhuan'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df_finance, x='doanh_thu_musd', y='loi_nhuan_musd', ax=axes[0]).set_title("Trước chuẩn hóa")
sns.scatterplot(data=df_fin_mm, x='doanh_thu', y='loi_nhuan', ax=axes[1]).set_title("Sau Min-Max")
sns.scatterplot(data=df_fin_zs, x='doanh_thu', y='loi_nhuan', ax=axes[2]).set_title("Sau Z-Score")
plt.show()

# THẢO LUẬN BÀI 3[cite: 24, 25]: 
# - Dữ liệu có ngoại lệ cực lớn (Doanh thu 10,000)[cite: 23]. 
# - Nên chọn Z-Score cho các mô hình hồi quy (Linear Regression) để tránh việc ngoại lệ làm lệch mô hình.

# =================================================================
# BÀI 4: NGƯỜI CHƠI TRỰC TUYẾN (Gaming Data) [cite: 26]
# =================================================================
print("\n--- Đang xử lý Bài 4 ---")
df_gaming = pd.read_csv('ITA105_Lab_3_Gaming.csv')

# Vẽ histogram so sánh phân phối điểm tích lũy [cite: 31]
plt.figure(figsize=(10, 5))
sns.histplot(df_gaming['diem_tich_luy'], kde=True)
plt.title("Phân phối điểm tích lũy của người chơi")
plt.show()

# THẢO LUẬN BÀI 4[cite: 36, 37]:
# - Một số người chơi cực kỳ "cày cuốc" (20,000 điểm) là ngoại lệ. 
# - Khi dùng KNN hoặc Clustering, Z-Score ổn hơn vì nó tính toán dựa trên độ lệch chuẩn, 
#   không bị ảnh hưởng quá nặng nề bởi khoảng cách cực đại như Min-Max.