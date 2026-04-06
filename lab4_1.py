import pandas as pd
# dùng để xử lý bảng dữ liệu
import numpy as np
# dùng cho toán học
import re
# xử lý chuỗi

from sklearn.preprocessing import LabelEncoder# mã hóa nhãn
# sklearn.preprocessing module cung cấp công cụ chuẩn hóa và mã hóa dữ liệu trước khi đưa vào mô hình học máy.
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
# dùng để học từ giống nhau

#2. danh sách stop word(từ không quan trọng)
stop_words = ["là", "và", "có", "rất", "thì", "một", "những", "các", "ờ"]

#3. hàm xử lý text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]","",text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return words

#bài 1
data1 = pd.DataFrame({
    "hotel":["A","B","C","D"],
    "review":[
        "Phòng rất sạch sẽ và đẹp",
        "Dịch vụ kém và bẩn",
        "Khách sạn sạch sẽ và yên tĩnh",
        "Phòng không sạch và rất ổn"
        ]
})

le = LabelEncoder()
data1["hotel_encoded"] = le.fit_transform(data1["hotel"])

# Xử lý text
data1["tokens"] = data1["review"].apply(clean_text)

print(data1)

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data1["review"])

print("\nTF-IDF matrix:")
print(tfidf_matrix.toarray())

# Word2Vec
model = Word2Vec(sentences=data1["tokens"], vector_size=50, window=2, min_count=1)

# Tìm từ giống "sạch"
print("\nTừ gần 'sạch':")
print(model.wv.most_similar("sạch", topn=5))

# Giải thích
print("\nKhi nào dùng TF-IDF?")
print("- Khi cần đếm từ quan trọng")

print("Khi nào dùng Word2Vec?")
print("- Khi cần hiểu nghĩa của từ")


# =========================================
# ========== BÀI 2: BÌNH LUẬN TRẬN ĐẤU ==========
# =========================================

print("\n===== BÀI 2 =====")

data2 = pd.DataFrame({
    "team": ["A", "B", "A", "C"],
    "comment": [
        "Trận đấu rất xuất sắc",
        "Chơi tệ và chậm",
        "Cầu thủ đá rất hay",
        "Phong độ xuất sắc"
    ]
})

# Encoding
data2["team_encoded"] = le.fit_transform(data2["team"])

# Clean text
data2["tokens"] = data2["comment"].apply(clean_text)

# TF-IDF
tfidf2 = TfidfVectorizer()
tfidf_matrix2 = tfidf2.fit_transform(data2["comment"])

# Word2Vec
model2 = Word2Vec(sentences=data2["tokens"], vector_size=50, min_count=1)

print("\nTừ gần 'xuất':")
print(model2.wv.most_similar("xuất", topn=5))

print("\nSo sánh:")
print("- TF-IDF: chỉ đếm từ")
print("- Word2Vec: hiểu nghĩa tốt hơn")


# =========================================
# ========== BÀI 3: FEEDBACK NGƯỜI CHƠI ==========
# =========================================

print("\n===== BÀI 3 =====")

data3 = pd.DataFrame({
    "game": ["X", "Y", "X", "Z"],
    "feedback": [
        "Game rất đẹp và hay",
        "Game xấu và lag",
        "Đồ họa đẹp tuyệt vời",
        "Chơi không vui"
    ]
})

# Encoding
data3["game_encoded"] = le.fit_transform(data3["game"])

# Clean
data3["tokens"] = data3["feedback"].apply(clean_text)

# TF-IDF
tfidf3 = TfidfVectorizer()
tfidf_matrix3 = tfidf3.fit_transform(data3["feedback"])

# Word2Vec
model3 = Word2Vec(sentences=data3["tokens"], vector_size=50, min_count=1)

print("\nTừ gần 'đẹp':")
print(model3.wv.most_similar("đẹp", topn=5))

print("\nChọn gì để phân loại cảm xúc?")
print("→ Chọn Word2Vec vì hiểu nghĩa tốt hơn")


# =========================================
# ========== BÀI 4: REVIEW ALBUM ==========
# =========================================

print("\n===== BÀI 4 =====")

data4 = pd.DataFrame({
    "artist": ["A", "B", "A", "C"],
    "review": [
        "Album rất sáng tạo",
        "Nhạc chán và lặp lại",
        "Giai điệu sáng tạo và mới",
        "Không hay"
    ]
})

# Encoding
data4["artist_encoded"] = le.fit_transform(data4["artist"])

# Clean
data4["tokens"] = data4["review"].apply(clean_text)

# TF-IDF
tfidf4 = TfidfVectorizer()
tfidf_matrix4 = tfidf4.fit_transform(data4["review"])

# Word2Vec
model4 = Word2Vec(sentences=data4["tokens"], vector_size=50, min_count=1)

print("\nTừ gần 'sáng':")
print(model4.wv.most_similar("sáng", topn=5))


# =========================================
# KẾT LUẬN CHUNG , CÁC EM ĐỌC KỸ MẤY CÁI NÀY ĐỂ HIÊU KHÁI NUEEMJ NHÉ
# =========================================

print("""
KẾT LUẬN:

TF-IDF:
- Giống như đếm từ
- Từ nào xuất hiện nhiều → quan trọng

Word2Vec:
- Hiểu nghĩa của từ
- Ví dụ: "đẹp" gần với "xinh", "tốt"

=> Nếu chỉ cần đếm → TF-IDF
=> Nếu cần hiểu nghĩa → Word2Vec
""")