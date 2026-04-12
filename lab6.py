import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_MAP = {
    'bai1': 'apartment.jpg',
    'bai2': 'car.jpg',       
    'bai3': 'fruit.jpg',     
    'bai4': 'room.jpg'      
}

def load_and_resize(path):
    """Đọc ảnh và resize về chuẩn 224x224"""
    img = cv2.imread(path)
    if img is None:
        # Tạo ảnh trống nếu không tìm thấy file để tránh lỗi code
        return np.zeros((224, 224, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (224, 224))

def normalize(img):
    """Chuẩn hóa pixel về [0, 1]"""
    return img.astype(np.float32) / 255.0

# --- CÁC HÀM AUGMENTATION ---

def add_gaussian_noise(img):
    """Thêm nhiễu Gaussian (Dùng cho Bài 2)"""
    row, col, ch = img.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img + gauss.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness(img, factor):
    """Thay đổi độ sáng"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def rotate_img(img, angle):
    """Xoay ảnh một góc bất kỳ"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def process_lab6():
    # BÀI 1: CĂN HỘ (Resize, Flip, Rotate 15, Bright 20%, Grayscale, Normalize)
    print("Đang xử lý Bài 1...")
    img1 = load_and_resize(DATA_MAP['bai1'])
    aug1 = rotate_img(cv2.flip(img1, 1), 15)
    aug1 = adjust_brightness(aug1, 1.2)
    gray1 = cv2.cvtColor(aug1, cv2.COLOR_RGB2GRAY)
    norm1 = normalize(gray1)

    # BÀI 2: XE CỘ (Resize, Noise, Bright 15%, Rotate 10, Normalize)
    print("Đang xử lý Bài 2...")
    img2 = load_and_resize(DATA_MAP['bai2'])
    aug2 = add_gaussian_noise(img2)
    aug2 = rotate_img(adjust_brightness(aug2, 0.85), -10)
    norm2 = normalize(aug2)

    # BÀI 3: TRÁI CÂY (Resize, Flip, Zoom/Rotation, Grid 3x3)
    print("Đang xử lý Bài 3...")
    img3 = load_and_resize(DATA_MAP['bai3'])
    # Tạo danh sách 9 ảnh biến thể cho grid 3x3
    aug_list3 = []
    for i in range(9):
        tmp = rotate_img(img3, np.random.randint(-30, 30))
        if i % 2 == 0: tmp = cv2.flip(tmp, 0)
        aug_list3.append(normalize(tmp))

    # BÀI 4: PHÒNG (Resize, Rotate 15, Flip, Bright 20%, Grayscale, Normalize)
    print("Đang xử lý Bài 4...")
    img4 = load_and_resize(DATA_MAP['bai4'])
    aug4 = rotate_img(cv2.flip(img4, 1), -15)
    aug4 = adjust_brightness(aug4, 1.2)
    gray4 = cv2.cvtColor(aug4, cv2.COLOR_RGB2GRAY)
    norm4 = normalize(gray4)

    # --- HIỂN THỊ KẾT QUẢ ---
    
    # Hiển thị Bài 1, 2, 4
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img1); axes[0, 0].set_title("B1: Gốc")
    axes[1, 0].imshow(norm1, cmap='gray'); axes[1, 0].set_title("B1: Xử lý (Xám)")
    
    axes[0, 1].imshow(img2); axes[0, 1].set_title("B2: Gốc")
    axes[1, 1].imshow(norm2); axes[1, 1].set_title("B2: Xử lý (Nhiễu/Xoay)")
    
    axes[0, 2].imshow(img4); axes[0, 2].set_title("B4: Gốc")
    axes[1, 2].imshow(norm4, cmap='gray'); axes[1, 2].set_title("B4: Xử lý (Xám)")
    plt.tight_layout()
    plt.show()

    # Hiển thị Bài 3: Grid 3x3
    fig3, axes3 = plt.subplots(3, 3, figsize=(10, 10))
    fig3.suptitle("Bài 3: Augmentation Grid 3x3 - Trái cây")
    for i, ax in enumerate(axes3.flat):
        ax.imshow(aug_list3[i])
        ax.axis('off')
    plt.show()
    
if __name__ == "__main__":
    process_lab6()
    print("✅ Hoàn thành cập nhật Lab 6!")