import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score

# 0. CHUẨN BỊ DỮ LIỆU
# Đảm bảo file CSV nằm cùng thư mục với file code này
df = pd.read_csv('ITA105_Lab_7.csv')

# BÀI 1: PHÂN TÍCH DỮ LIỆU & KHÁM PHÁ PHÂN PHỐI (2đ)
print("--- BÀI 1: PHÂN TÍCH SKEWNESS ---")

# 1. Tính skewness và lập bảng top 10
numeric_cols = df.select_dtypes(include=[np.number]).columns
skewness_series = df[numeric_cols].skew().sort_values(ascending=False)
print("Top 10 cột lệch nhất:")
print(skewness_series.head(10))

# 2. Vẽ biểu đồ cho 3 cột lệch nhất
top_3_cols = skewness_series.index[:3]
plt.figure(figsize=(18, 5))
for i, col in enumerate(top_3_cols):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Phân phối {col}\nSkewness: {df[col].skew():.2f}')
plt.tight_layout()
plt.show()

print("\n[Phân tích Bài 1]:")
print("- Xu hướng: Các biến như LotArea và SalePrice có xu hướng lệch dương (đuôi dài bên phải).")
print("- Outlier: Xuất hiện nhiều giá trị cực lớn tách biệt hẳn khỏi đám đông.")
print("- Nguyên nhân: Do đặc thù kinh tế, số lượng nhà giá rẻ/diện tích nhỏ luôn chiếm đa số.")
print("- Tác động: Skewness cao làm Linear Regression dự báo kém chính xác vì bị nhiễu bởi outlier.")

# BÀI 2: BIẾN ĐỔI DỮ LIỆU NÂNG CAO
print("\n--- BÀI 2: KỸ THUẬT BIẾN ĐỔI ---")

# Chọn cột theo yêu cầu: 2 dương (SalePrice, LotArea) và 1 âm (NegSkewIncome)
col1, col2, col3 = 'SalePrice', 'LotArea', 'NegSkewIncome'

# Áp dụng 3 kỹ thuật
# 1. np.log()
df[f'{col1}_log'] = np.log(df[col1])

# 2. Box-Cox (Tìm lambda tối ưu)
df[f'{col2}_boxcox'], lmbda = stats.boxcox(df[col2])
print(f"Lambda tối ưu cho {col2}: {lmbda:.4f}")

# 3. PowerTransformer (Yeo-Johnson) cho cột có giá trị âm
pt = PowerTransformer(method='yeo-johnson')
df[f'{col3}_pt'] = pt.fit_transform(df[[col3]])

# Lập bảng so sánh 
comparison_data = {
    'Cột': [col1, col2, col3],
    'Skew trước': [df[col1].skew(), df[col2].skew(), df[col3].skew()],
    'Skew sau Log': [df[f'{col1}_log'].skew(), np.nan, np.nan],
    'Skew sau Box-Cox': [np.nan, df[f'{col2}_boxcox'].skew(), np.nan],
    'Skew sau Power': [np.nan, np.nan, df[f'{col3}_pt'].skew()]
}
df_comp = pd.DataFrame(comparison_data)
print(df_comp)

# Vẽ biểu đồ trước - sau
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
cols_pair = [(col1, f'{col1}_log'), (col2, f'{col2}_boxcox'), (col3, f'{col3}_pt')]
for i, (orig, trans) in enumerate(cols_pair):
    sns.histplot(df[orig], kde=True, ax=axes[i, 0], color='red')
    axes[i, 0].set_title(f"Trước: {orig}")
    sns.histplot(df[trans], kde=True, ax=axes[i, 1], color='green')
    axes[i, 1].set_title(f"Sau biến đổi: {trans}")
plt.tight_layout()
plt.show()

# BÀI 3: ỨNG DỤNG VÀO MÔ HÌNH HÓA
print("\n--- BÀI 3: MÔ HÌNH HÓA ---")

# Chuẩn bị X, y
features = ['LotArea', 'HouseAge', 'MixedFeature', 'Rooms']
X = df[features]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Version A: Gốc
model_a = LinearRegression().fit(X_train, y_train)
pred_a = model_a.predict(X_test)

# Version B: Log biến mục tiêu
model_b = LinearRegression().fit(X_train, np.log(y_train))
pred_b_log = model_b.predict(X_test)
pred_b_final = np.exp(pred_b_log) # Dịch ngược

# Version C: PowerTransformer cho features
pt_feat = PowerTransformer()
X_train_c = pt_feat.fit_transform(X_train)
X_test_c = pt_feat.transform(X_test)
model_c = LinearRegression().fit(X_train_c, y_train)
pred_c = model_c.predict(X_test_c)

# Đánh giá
def get_metrics(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

rmse_a, r2_a = get_metrics(y_test, pred_a)
rmse_b, r2_b = get_metrics(y_test, pred_b_final)
rmse_c, r2_c = get_metrics(y_test, pred_c)

print(f"Version A (Gốc)      - RMSE: {rmse_a:.2f}, R2: {r2_a:.4f}")
print(f"Version B (Log Target)- RMSE: {rmse_b:.2f}, R2: {r2_b:.4f}")
print(f"Version C (PT Feats)  - RMSE: {rmse_c:.2f}, R2: {r2_c:.4f}")

# BÀI 4: ỨNG DỤNG NGHIỆP VỤ
print("\n--- BÀI 4: INSIGHT NGHIỆP VỤ ---")

# Biểu đồ so sánh cho SalePrice
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], kde=True, color='orange')
plt.title("Giá nhà gốc (Khó nhìn phân khúc thấp)")
plt.subplot(1, 2, 2)
sns.histplot(np.log(df['SalePrice']), kde=True, color='purple')
plt.title("Chỉ số Log-Price (Dễ phân loại khách hàng)")
plt.show()

# Tạo metric mới
df['log_price_index'] = np.log(df['SalePrice'])

print("\n[Insight cho nhà quản lý]:")
print("1. Tại sao cần biến đổi? Giúp chúng ta không bị 'lóa mắt' bởi vài căn biệt thự quá đắt mà bỏ quên thị trường đại chúng.")
print("2. Ứng dụng: Chỉ số 'log_price_index' giúp phân nhóm khách hàng theo tỉ lệ phần trăm ngân sách thay vì số tiền tuyệt đối.")
print("3. Khuyến nghị: Sử dụng mô hình Version C để định giá nhà ổn định hơn cho khách hàng mua để ở.")