import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class CircleObjectDetector:
    def __init__(self):
        """Khởi tạo detector với các tham số mặc định"""
        self.min_radius = 10
        self.max_radius = 200
        self.dp = 1
        self.min_dist = 50
        self.param1 = 50
        self.param2 = 30

    def preprocess_image(self, image):
        """
        Tiền xử lý ảnh để tăng độ chính xác nhận diện
        """
        # Chuyển sang ảnh xám
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Áp dụng Gaussian blur để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Tăng cường độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced

    def detect_circles(self, image, show_steps=False):
        """
        Nhận diện các vật thể hình tròn trong ảnh
        """
        original = image.copy()

        # Tiền xử lý ảnh
        processed = self.preprocess_image(image)

        # Sử dụng HoughCircles để phát hiện hình tròn
        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        results = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Vẽ các hình tròn được phát hiện
            result_image = original.copy()

            for i, (x, y, r) in enumerate(circles):
                # Vẽ đường viền hình tròn
                cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
                # Vẽ tâm
                cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)
                # Thêm nhãn
                cv2.putText(result_image, f'#{i + 1}', (x - 10, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                results.append({
                    'id': i + 1,
                    'center': (x, y),
                    'radius': r,
                    'area': np.pi * r * r
                })
        else:
            result_image = original.copy()

        if show_steps:
            self.show_detection_steps(original, processed, result_image)

        return result_image, results

    def show_detection_steps(self, original, processed, result):
        """
        Hiển thị các bước xử lý ảnh
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Ảnh gốc
        if len(original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Ảnh gốc')
        axes[0].axis('off')

        # Ảnh sau tiền xử lý
        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title('Ảnh sau tiền xử lý')
        axes[1].axis('off')

        # Kết quả nhận diện
        if len(result.shape) == 3:
            axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            axes[2].imshow(result, cmap='gray')
        axes[2].set_title('Kết quả nhận diện')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    def adjust_parameters(self, min_radius=None, max_radius=None,
                          min_dist=None, param1=None, param2=None):
        """
        Điều chỉnh các tham số để tối ưu hóa nhận diện
        """
        if min_radius is not None:
            self.min_radius = min_radius
        if max_radius is not None:
            self.max_radius = max_radius
        if min_dist is not None:
            self.min_dist = min_dist
        if param1 is not None:
            self.param1 = param1
        if param2 is not None:
            self.param2 = param2

    def analyze_results(self, results):
        """
        Phân tích kết quả nhận diện
        """
        if not results:
            print("Không phát hiện vật thể nào!")
            return

        print(f"Đã phát hiện {len(results)} vật thể hình tròn:")
        print("-" * 50)

        total_area = 0
        radii = []

        for obj in results:
            print(f"Vật thể #{obj['id']}:")
            print(f"  - Tọa độ tâm: {obj['center']}")
            print(f"  - Bán kính: {obj['radius']} pixels")
            print(f"  - Diện tích: {obj['area']:.1f} pixels²")

            total_area += obj['area']
            radii.append(obj['radius'])
            print()

        print("Thống kê tổng quan:")
        print(f"  - Tổng diện tích: {total_area:.1f} pixels²")
        print(f"  - Bán kính trung bình: {np.mean(radii):.1f} pixels")
        print(f"  - Bán kính nhỏ nhất: {min(radii)} pixels")
        print(f"  - Bán kính lớn nhất: {max(radii)} pixels")


def main():
    """
    Hàm chính để demo chương trình
    """
    # Khởi tạo detector
    detector = CircleObjectDetector()

    # Tạo ảnh mẫu với các hình tròn
    def create_sample_image():
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img.fill(50)  # Nền xám tối

        # Vẽ một số hình tròn mô phỏng cà chua
        circles_data = [
            ((150, 150), 40, (0, 0, 200)),  # Đỏ
            ((350, 150), 35, (0, 0, 180)),  # Đỏ đậm hơn
            ((450, 200), 45, (0, 0, 220)),  # Đỏ tươi
            ((200, 280), 30, (0, 0, 160)),  # Đỏ thẫm
            ((400, 320), 38, (0, 0, 190)),  # Đỏ vừa
        ]

        for center, radius, color in circles_data:
            cv2.circle(img, center, radius, color, -1)
            # Thêm một chút highlight
            cv2.circle(img, (center[0] - 10, center[1] - 10), radius // 4,
                       (color[0] + 50, color[1] + 50, min(255, color[2] + 50)), -1)

        # Thêm một chút nhiễu
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

        return img

    # Tạo ảnh mẫu
    sample_image = create_sample_image()

    print("=== CHƯƠNG TRÌNH NHẬN DIỆN VẬT THỂ HÌNH TRÒN ===")
    print("Ví dụ: Nhận diện cà chua")
    print()

    # Điều chỉnh tham số cho cà chua
    detector.adjust_parameters(
        min_radius=20,
        max_radius=60,
        min_dist=40,
        param1=50,
        param2=25
    )

    # Thực hiện nhận diện
    result_image, results = detector.detect_circles(sample_image, show_steps=True)

    # Phân tích kết quả
    detector.analyze_results(results)

    # Lưu kết quả
    cv2.imwrite('detected_circles.jpg', result_image)
    print("\nĐã lưu kết quả vào file 'detected_circles.jpg'")


def detect_from_file(image_path):
    """
    Nhận diện từ file ảnh
    """
    detector = CircleObjectDetector()

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return

    # Điều chỉnh tham số tùy theo loại ảnh
    detector.adjust_parameters(min_radius=10, max_radius=100, min_dist=30)

    # Thực hiện nhận diện
    result_image, results = detector.detect_circles(image, show_steps=True)

    # Phân tích kết quả
    detector.analyze_results(results)

    # Lưu kết quả
    output_path = f"result_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\nĐã lưu kết quả vào {output_path}")


if __name__ == "__main__":
    # Chạy demo với ảnh mẫu
    # main()

    # Uncomment dòng dưới để test với ảnh thật
    detect_from_file("best4_jpeg.rf.3df643a002bf5a83cd6ef2b3ac572eae.jpg")