# =========================================
# LAB 8 - FULL PIPELINE (FINAL VERSION) - LOCAL PC
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# =========================================
# 0. Load dữ liệu (Chạy trên máy tính/Local)
# =========================================
# BẠN HÃY SỬA TÊN TỆP VÀO ĐÂY (ví dụ: data.csv)
FILE_PATH = r"C:\Users\ACER\OneDrive\Desktop\tien_xu_ly_du_lieu\lab8\data.csv"

try:
    df = pd.read_csv(FILE_PATH)
    print("Data shape:", df.shape)
    print(df.head())
except FileNotFoundError:
    raise Exception(f"Không tìm thấy file tại {FILE_PATH}. Vui lòng kiểm tra lại tên tệp!")


# =========================================
# 2. CLEAN DATA BAN ĐẦU
# =========================================
# ép numeric an toàn
for col in df.select_dtypes(include=np.number).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# =========================================
# 3. CUSTOM TRANSFORMERS
# =========================================
class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lower = np.percentile(X, 1, axis=0)
        self.upper = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower, self.upper)


class TimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        dt = pd.to_datetime(X.iloc[:,0], errors='coerce')
        return pd.DataFrame({
            "month": dt.dt.month,
            "quarter": dt.dt.quarter,
            "year": dt.dt.year
        })


def clean_text(x):
    return x.fillna("").astype(str)

def safe_log(x):
    return np.log1p(np.maximum(x, 0))


# =========================================
# 4. PHÂN LOẠI CỘT
# =========================================
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

text_col = None
time_col = None

for col in cat_cols:
    if "desc" in col.lower():
        text_col = col
    if "date" in col.lower():
        time_col = col

if text_col in cat_cols:
    cat_cols.remove(text_col)

if time_col in cat_cols:
    cat_cols.remove(time_col)

print("Num:", num_cols)
print("Cat:", cat_cols)
print("Text:", text_col)
print("Time:", time_col)


# =========================================
# 5. PIPELINE CHO PREPROCESS (FULL DATA)
# =========================================
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("outlier", OutlierClipper()),
    ("scale", StandardScaler()),
    ("log", FunctionTransformer(safe_log))
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

text_pipeline = Pipeline([
    ("clean", FunctionTransformer(clean_text)),
    ("tfidf", TfidfVectorizer(max_features=100))
])

time_pipeline = Pipeline([
    ("extract", TimeFeatures()),
    ("impute", SimpleImputer(strategy="most_frequent"))
])

transformers = [
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
]

if text_col:
    transformers.append(("text", text_pipeline, text_col))

if time_col:
    transformers.append(("time", time_pipeline, [time_col]))

preprocessor = ColumnTransformer(transformers)

pipeline = Pipeline([
    ("prep", preprocessor)
])


# =========================================
# BÀI 1: SMOKE TEST + FEATURE NAMES
# =========================================
sample = df.head(10)
out = pipeline.fit_transform(sample)
print("Smoke test OK:", out.shape)

def get_feature_names(ct):
    features = []
    for name, trans, cols in ct.transformers_:
        if name == "num":
            features.extend(cols)
        elif name == "cat":
            features.extend(trans.named_steps["onehot"].get_feature_names_out(cols))
        elif name == "text":
            features.extend(trans.named_steps["tfidf"].get_feature_names_out())
        elif name == "time":
            features.extend(["month", "quarter", "year"])
    return features

pipeline.fit(df)
print("Total features:", len(get_feature_names(preprocessor)))


# =========================================
# BÀI 2: TEST PIPELINE
# =========================================
df_full = df.copy()

df_missing = df.copy()
for col in num_cols:
    df_missing.loc[:int(len(df)*0.8), col] = np.nan

df_skew = df.copy()
df_skew[num_cols] *= 1000

df_unseen = df.copy()
if cat_cols:
    df_unseen[cat_cols[0]] = "new_value"

df_wrong = df.copy()
if num_cols:
    c = num_cols[0]
    df_wrong[c] = "abc"
    df_wrong[c] = pd.to_numeric(df_wrong[c], errors='coerce')

test_sets = {
    "full": df_full,
    "missing": df_missing,
    "skew": df_skew,
    "unseen": df_unseen,
    "wrong": df_wrong
}

for name, data in test_sets.items():
    print("\n---", name, "---")
    try:
        out = pipeline.fit_transform(data)
        print("OK | shape:", out.shape)
    except Exception as e:
        print("ERROR:", e)


# =========================================
# SO SÁNH PHÂN PHỐI
# =========================================
if num_cols:
    col = num_cols[0]

    plt.hist(df[col].dropna(), bins=30)
    plt.title("Before")
    plt.show()

    processed = pipeline.fit_transform(df)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    plt.hist(processed[:,0], bins=30)
    plt.title("After")
    plt.show()


# =========================================
# BÀI 3: MODEL + CV (FIX CHUẨN)
# =========================================
target = num_cols[0]

X = df.drop(columns=[target])
y = df[target]

# cập nhật lại cột (QUAN TRỌNG)
num_cols_model = [c for c in num_cols if c != target]
cat_cols_model = [c for c in cat_cols if c in X.columns]

preprocessor_model = ColumnTransformer([
    ("num", num_pipeline, num_cols_model),
    ("cat", cat_pipeline, cat_cols_model)
] + (
    [("text", text_pipeline, text_col)] if text_col else []
) + (
    [("time", time_pipeline, [time_col])] if time_col else []
))

pipe_lr = Pipeline([
    ("prep", preprocessor_model),
    ("model", LinearRegression())
])

pipe_rf = Pipeline([
    ("prep", preprocessor_model),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

for name, model in [("LR", pipe_lr), ("RF", pipe_rf)]:
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(name, "R2:", scores.mean(), "| Var:", scores.var())


# =========================================
# SO SÁNH MANUAL
# =========================================
X_proc = preprocessor_model.fit_transform(X)
if hasattr(X_proc, "toarray"):
    X_proc = X_proc.toarray()

lr = LinearRegression()
print("Manual R2:", cross_val_score(lr, X_proc, y, cv=5).mean())


# =========================================
# BÀI 4: INFERENCE (Dự đoán tệp mới)
# =========================================
final_model = pipe_rf.fit(X, y)

print("\n--- Đang tiến hành dự đoán trên tệp dữ liệu mới ---")

# # BẠN HÃY SỬA TÊN TỆP MỚI DÙNG ĐỂ TEST VÀO ĐÂY
# NEW_FILE_PATH = r"C:\Users\ACER\OneDrive\Desktop\tien_xu_ly_du_lieu\lab8\ITA105_Lab_8_House_images"

# try:
#     new_df = pd.read_csv(NEW_FILE_PATH)
#     preds = final_model.predict(new_df)
#     print("Predictions:", preds[:10])
# except FileNotFoundError:
#     print(f"Lỗi: Không tìm thấy tệp dữ liệu test tại {NEW_FILE_PATH}. Vui lòng cập nhật tên tệp!")
# except ValueError as e:
#     print("Lỗi về dữ liệu (có thể tệp mới không khớp cấu trúc với tệp huấn luyện):", e)