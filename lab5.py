import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# Cấu hình hiển thị
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def smart_load(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return None
    df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    return df

def find_column(df, keywords):
    for col in df.columns:
        if any(key.lower() in col.lower() for key in keywords):
            return col
    return None

# ==========================================
# BÀI 1: DOANH THU SIÊU THỊ
# ==========================================
print("\n--- 🛒 Đang xử lý Bài 1 ---")
df1 = smart_load('ITA105_Lab_5_Supermarket.csv')
if df1 is not None:
    date_col = find_column(df1, ['date', 'ngày'])
    rev_col = find_column(df1, ['revenue', 'doanh thu'])
    
    df1[date_col] = pd.to_datetime(df1[date_col])
    df1.set_index(date_col, inplace=True)
    
    # 1. Xử lý missing values
    df1[rev_col] = df1[rev_col].interpolate(method='linear')
    
    # 2. Tạo đặc trưng (Yêu cầu của Lab)
    df1['Year'] = df1.index.year
    df1['Month'] = df1.index.month
    df1['Quarter'] = df1.index.quarter
    df1['DayOfWeek'] = df1.index.dayofweek
    df1['IsWeekend'] = df1.index.dayofweek >= 5 # 5 là Thứ 7, 6 là CN
    
    # 3. Vẽ biểu đồ doanh thu theo tháng
    df1.resample('M')[rev_col].sum().plot(marker='o', title='Bài 1: Tổng doanh thu theo tháng')
    plt.ylabel('Doanh thu')
    plt.show()

# ==========================================
# BÀI 2: LƯU LƯỢNG TRUY CẬP WEBSITE
# ==========================================
print("\n--- 🌐 Đang xử lý Bài 2 ---")
df2 = smart_load('ITA105_Lab_5_Web_traffic.csv')
if df2 is not None:
    time_col = find_column(df2, ['datetime', 'timestamp'])
    # Cập nhật từ khóa 'visits' để khớp với file CSV
    traffic_col = find_column(df2, ['visits', 'traffic', 'lượt truy cập'])
    
    df2[time_col] = pd.to_datetime(df2[time_col])
    df2.set_index(time_col, inplace=True)
    
    # 1. Resample hourly và nội suy
    df2 = df2.resample('H').mean().interpolate(method='linear')
    
    # 2. Tạo đặc trưng giờ và ngày trong tuần
    df2['Hour'] = df2.index.hour
    df2['DayOfWeek'] = df2.index.day_name()
    
    # 3. Vẽ traffic trung bình theo giờ
    df2.groupby('Hour')[traffic_col].mean().plot(kind='bar', color='skyblue', title='Bài 2: Traffic trung bình theo giờ')
    plt.show()

# ==========================================
# BÀI 3: GIÁ CỔ PHIẾU
# ==========================================
print("\n--- 📈 Đang xử lý Bài 3 ---")
df3 = smart_load('ITA105_Lab_5_Stock.csv')
if df3 is not None:
    date_col = find_column(df3, ['date', 'ngày'])
    close_col = find_column(df3, ['close', 'đóng cửa'])
    
    df3[date_col] = pd.to_datetime(df3[date_col])
    df3.set_index(date_col, inplace=True)
    
    # 1. Điền giá trị thiếu bằng Forward Fill (giá ngày trước đó)
    df3[close_col] = df3[close_col].ffill()
    
    # 2. Tạo Moving Average
    df3['MA7'] = df3[close_col].rolling(window=7).mean()
    df3['MA30'] = df3[close_col].rolling(window=30).mean()
    
    # 3. Vẽ biểu đồ xu hướng
    df3[[close_col, 'MA7', 'MA30']].plot(title='Bài 3: Xu hướng giá cổ phiếu (MA7 & MA30)')
    plt.show()
    
    # 4. Tính mùa vụ theo tháng (Yêu cầu bổ sung)
    df3.groupby(df3.index.month)[close_col].mean().plot(kind='bar', title='Bài 3: Giá trung bình theo tháng')
    plt.show()

# ==========================================
# BÀI 4: SẢN XUẤT CÔNG NGHIỆP
# ==========================================
print("\n--- 🏭 Đang xử lý Bài 4 ---")
df4 = smart_load('ITA105_Lab_5_Production.csv')
if df4 is not None:
    date_col = find_column(df4, ['week_start', 'date'])
    prod_col = find_column(df4, ['production', 'sản xuất'])
    
    df4[date_col] = pd.to_datetime(df4[date_col])
    df4.set_index(date_col, inplace=True)
    
    # 1. Xử lý missing values
    df4[prod_col] = df4[prod_col].ffill()
    
    # 2. Tạo đặc trưng tuần, quý, năm
    df4['Week'] = df4.index.isocalendar().week
    df4['Quarter'] = df4.index.quarter
    df4['Year'] = df4.index.year
    
    # 3. Phân tích Decomposition
    # Vì dữ liệu theo tuần, dùng period=52 (số tuần trong 1 năm)
    result = seasonal_decompose(df4[prod_col], model='additive', period=52)
    result.plot()
    plt.suptitle('Bài 4: Phân tích thành phần dữ liệu sản xuất', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\n✅ Hoàn thành Lab 5!")