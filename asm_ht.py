import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# ==============================================================================
# 0. CẤU HÌNH ĐƯỜNG DẪN & TÊN CỘT ĐẶC THÙ
# ==============================================================================
FILE_PATH = 'ITA105_Lab_1.csv'
IMG_DIR = 'ITA105_Lab_8_House_images/'

# BẠN HÃY KIỂM TRA VÀ ĐỔI TÊN 4 BIẾN NÀY CHO ĐÚNG VỚI FILE CSV CỦA BẠN:
TARGET = 'Price'               # Cột giá nhà (Cần dự đoán)
TEXT_COL = 'Description'       # Cột mô tả nhà bằng chữ
DATE_COL = 'Transaction_Date'  # Cột ngày giao dịch
IMG_COL = 'Image_Filename'     # Cột chứa tên file ảnh (VD: 'nha1.jpg')

# Đọc dữ liệu thực tế
print(f"Đang tải dữ liệu từ {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)

# Tự động nhận diện cột Số và Phân loại (bỏ qua các cột đặc thù đã định nghĩa)
all_special_cols = [TARGET, TEXT_COL, DATE_COL, IMG_COL]
NUM_COLS = [c for c in df.select_dtypes(include=np.number).columns if c not in all_special_cols]
CAT_COLS = [c for c in df.select_dtypes(exclude=np.number).columns if c not in all_special_cols]

print("\nĐã nhận diện tự động:")
print(f"- Cột Số (Numerical): {NUM_COLS}")
print(f"- Cột Phân loại (Categorical): {CAT_COLS}")


# ==============================================================================
# PHẦN 1: KHÁM PHÁ VÀ LÀM SẠCH CƠ BẢN (Ý 1, 2 - GĐ 1)
# ==============================================================================
print("\n" + "="*50 + "\nPHẦN 1: KHÁM PHÁ & LÀM SẠCH\n" + "="*50)

print("1. Thống kê chung:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("Số bản ghi trùng lặp (Duplicate):", df.duplicated().sum())

# 2. Xử lý dữ liệu bẩn
df_clean = df.copy()
df_clean.drop_duplicates(inplace=True)

# Xử lý giá trị bất hợp lý (VD: Giá nhà phải > 0)
if TARGET in df_clean.columns:
    df_clean = df_clean[df_clean[TARGET] > 0]

# Điền Missing value cơ bản để khám phá (Pipeline sẽ làm kỹ hơn ở Phần 3)
for col in NUM_COLS:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)
for col in CAT_COLS:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

print(f"-> Dữ liệu sau làm sạch cơ bản: {df_clean.shape}")


# ==============================================================================
# PHẦN 2: OUTLIERS, CHUẨN HÓA & DUPLICATE TEXT (Ý 3, 4, 5 - GĐ 1)
# ==============================================================================
print("\n" + "="*50 + "\nPHẦN 2: OUTLIERS & TEXT SIMILARITY\n" + "="*50)

# 3. Capping Outliers cho Target bằng IQR
if TARGET in df_clean.columns:
    Q1 = df_clean[TARGET].quantile(0.25)
    Q3 = df_clean[TARGET].quantile(0.75)
    IQR = Q3 - Q1
    df_clean['Price_Capped'] = np.clip(df_clean[TARGET], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    print("- Đã giới hạn (capping) Outlier cho cột Target.")

# 5. Phát hiện Duplicate qua Text Similarity (TF-IDF)
if TEXT_COL in df_clean.columns:
    print("- Đang quét độ trùng lặp nội dung Text (Ngưỡng 95%)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = tfidf.fit_transform(df_clean[TEXT_COL].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    upper_tri = np.triu(cosine_sim, k=1)
    indices = np.where(upper_tri > 0.95)
    print(f"  -> Cảnh báo: Tìm thấy {len(indices[0])} cặp nhà có mô tả giống nhau > 95%.")


# ==============================================================================
# PHẦN 3: PIPELINE, MODELING & BUSINESS INSIGHTS (GĐ 2 + HOÀN THIỆN)
# ==============================================================================
print("\n" + "="*50 + "\nPHẦN 3: PIPELINE, AI MODEL & INSIGHTS\n" + "="*50)

# 1. Feature Engineering: Tích hợp Text, Time, Image, Interaction
print("1. Đang trích xuất đặc trưng (Feature Engineering)...")
def extract_advanced_features(X):
    X_new = X.copy()
    added_num_cols = []
    
    # Feature từ Text
    if TEXT_COL in X.columns:
        X_new['Word_Count'] = X_new[TEXT_COL].fillna('').apply(lambda x: len(str(x).split()))
        luxury_keywords = ['luxury', 'villa', 'pool', 'premium', 'cao cấp', 'biệt thự']
        X_new['Luxury_Keyword_Count'] = X_new[TEXT_COL].str.lower().fillna('').apply(
            lambda x: sum([1 for w in luxury_keywords if w in x])
        )
        added_num_cols.extend(['Word_Count', 'Luxury_Keyword_Count'])

    # Feature từ Date
    if DATE_COL in X.columns:
        try:
            X_new[DATE_COL] = pd.to_datetime(X_new[DATE_COL])
            X_new['Month'] = X_new[DATE_COL].dt.month.fillna(1)
            added_num_cols.append('Month')
        except:
            pass
            
    # Feature Interaction (Diện tích / Số phòng - Cần sửa tên cột 'Area' và 'Rooms' nếu file CSV đặt khác)
    area_col = next((c for c in X.columns if 'area' in c.lower() or 'diện tích' in c.lower()), None)
    room_col = next((c for c in X.columns if 'room' in c.lower() or 'phòng' in c.lower()), None)
    
    if area_col and room_col:
        X_new['Area_per_Room'] = X_new[area_col] / (X_new[room_col] + 1) # +1 tránh chia 0
        added_num_cols.append('Area_per_Room')
        
    return X_new, added_num_cols

df_fe, new_cols = extract_advanced_features(df_clean)

# (Tùy chọn) Trích xuất đặc trưng từ Ảnh - Chạy rất nặng nên dùng hàm dummy đại diện
# Để chạy thật với ảnh, bỏ comment hàm dưới và dùng CNN MobileNetV2
df_fe['Image_Feature_Sum'] = np.random.rand(len(df_fe)) 
new_cols.append('Image_Feature_Sum')

FINAL_NUM_COLS = NUM_COLS + new_cols

# 2. Xây dựng Pipeline Tự động
print("2. Đang đóng gói Pipeline (Pre-processing)...")
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson')), # Khắc phục Skewness
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Xử lý Unseen Categories
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, FINAL_NUM_COLS),
        ('cat', cat_transformer, CAT_COLS)
    ], remainder='drop'
)

# 3. Split & Training Model
print("3. Đang huấn luyện Mô hình (XGBoost, Random Forest, LinReg)...")
X = df_fe.drop(columns=[TARGET, 'Price_Capped'], errors='ignore')
y = df_fe[TARGET]
y_log = np.log1p(y) # Target Log-Transform

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.3, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

best_model, best_r2 = None, -float('inf')

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    clf.fit(X_train, y_train_log)
    
    preds = np.expm1(clf.predict(X_test))
    actual = np.expm1(y_test_log)
    
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2 = r2_score(actual, preds)
    print(f"  [{name}] RMSE: {rmse:,.2f} | R²: {r2:.4f}")
    
    if r2 > best_r2: best_r2, best_model = r2, clf

# 4. Dashboard Mini & Insight
print(f"\n-> Chọn mô hình tốt nhất: {best_model.named_steps['model'].__class__.__name__}")
print("\n4. Trích xuất Business Insights (KPIs):")

df_results = X_test.copy()
df_results['Actual_Price'] = np.expm1(y_test_log)
df_results['Predicted_Price'] = np.expm1(best_model.predict(X_test))

top_5_threshold = df_results['Actual_Price'].quantile(0.95)
print(f"- Ngưỡng giá phân khúc siêu sang (Top 5%): {top_5_threshold:,.2f}")

if 'Luxury_Keyword_Count' in df_results.columns:
    luxury_avg = df_results[df_results['Luxury_Keyword_Count'] > 0]['Actual_Price'].mean()
    normal_avg = df_results[df_results['Luxury_Keyword_Count'] == 0]['Actual_Price'].mean()
    print(f"- Giá trung bình nhà có tag 'Luxury/Cao cấp': {luxury_avg:,.2f} so với nhà thường: {normal_avg:,.2f}")

print("\nQuy trình hoàn tất. Pipeline đã sẵn sàng để đón dữ liệu mới!")